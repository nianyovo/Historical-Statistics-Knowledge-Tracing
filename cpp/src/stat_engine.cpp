#include "hsf/stat_engine.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include "hsf/thread_pool.hpp"

namespace hsf {
namespace {
void ValidateFlat(const Interaction* rows, std::size_t num_rows,
                  const std::int64_t* offsets, std::size_t num_seqs) {
  if (offsets == nullptr || (num_rows > 0 && rows == nullptr) ||
      num_rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      offsets[0] != 0 || offsets[num_seqs] != static_cast<std::int64_t>(num_rows)) {
    throw std::invalid_argument("invalid buffers or offsets");
  }
  for (std::size_t s = 0; s < num_seqs; ++s) {
    if (offsets[s] < 0 || offsets[s + 1] < offsets[s] ||
        offsets[s + 1] > static_cast<std::int64_t>(num_rows)) {
      throw std::invalid_argument("invalid offset ordering");
    }
    for (auto i = offsets[s]; i < offsets[s + 1]; ++i) {
      const auto& row = rows[i];
      if (row.user_id != rows[offsets[s]].user_id || row.item_id == 0 ||
          row.concept_id == 0 || (row.correct != 0 && row.correct != 1) ||
          !std::isfinite(row.timestamp_sec) || !std::isfinite(row.resp_time_sec) ||
          row.resp_time_sec < 0.0) {
        throw std::invalid_argument("invalid interaction");
      }
    }
  }
}

void ValidateOutput(std::size_t n, std::size_t threads, float* out,
                    std::size_t size) {
  if (threads == 0 || n > std::numeric_limits<std::size_t>::max() / kNumStatFeatures ||
      size < n * kNumStatFeatures || (n > 0 && out == nullptr)) {
    throw std::invalid_argument("invalid output capacity or threads");
  }
}

std::pair<std::size_t, std::size_t> DenseLimits(const Interaction* rows, std::size_t count) {
  std::size_t item_limit = 0;
  std::size_t concept_limit = 0;
  // Dense training IDs are <= row count. Sparse, very large C-API IDs must
  // not force an allocation proportional to the largest integer key.
  for (std::size_t i = 0; i < count; ++i) {
    if (rows[i].item_id > 0 && static_cast<std::size_t>(rows[i].item_id) <= count) {
      item_limit = std::max(item_limit, static_cast<std::size_t>(rows[i].item_id));
    }
    if (rows[i].concept_id > 0 && static_cast<std::size_t>(rows[i].concept_id) <= count) {
      concept_limit = std::max(concept_limit, static_cast<std::size_t>(rows[i].concept_id));
    }
  }
  return {item_limit, concept_limit};
}

void ParallelSequences(std::size_t count, std::size_t threads,
                       const std::function<void(std::size_t)>& function) {
  const auto workers = std::min(count, threads);
  if (workers <= 1) {
    for (std::size_t s = 0; s < count; ++s) { function(s); }
    return;
  }
  ThreadPool pool(workers);
  std::vector<std::future<void>> futures;
  for (std::size_t w = 0; w < workers; ++w) {
    const auto begin = (count / workers) * w + std::min(w, count % workers);
    const auto end = begin + count / workers + (w < count % workers ? 1 : 0);
    futures.push_back(pool.Submit([&, begin, end] {
      for (std::size_t s = begin; s < end; ++s) { function(s); }
    }));
  }
  // Drain every task before references captured by workers leave scope.
  std::exception_ptr error;
  for (auto& future : futures) {
    try { future.get(); } catch (...) { error = std::current_exception(); }
  }
  if (error) { std::rethrow_exception(error); }
}
}  // namespace

void Aggregate::Add(const Interaction& row) {
  global.Add(row);
  items.Add(row.item_id, row);
  concepts.Add(row.concept_id, row);
}

std::array<float, kNumStatFeatures> StudentState::Extract(
    const Interaction& row, const StatConfig& cfg, double mean,
    double item_acc, double concept_acc, double item_rt) const {
  const auto it = concepts_.find(row.concept_id);
  const auto state = it == concepts_.end() ? ConceptState{} : it->second;
  double gap = 0.0;
  if (state.count > 0) {
    gap = cfg.has_real_timestamp
        ? std::log1p(std::clamp(row.timestamp_sec - state.last_timestamp,
                                0.0, 2592000.0)) / 15.0
        : std::log1p(static_cast<double>(std::min<std::int64_t>(
              count_ - state.last_step, 500))) / 6.5;
  }
  return {static_cast<float>(item_acc), static_cast<float>(concept_acc),
          static_cast<float>(cfg.has_real_resp_time ? std::log1p(item_rt) / 6.0 : 0.0),
          static_cast<float>((correct_ + cfg.user_alpha * mean) / (count_ + cfg.user_alpha)),
          static_cast<float>(std::log1p(static_cast<double>(count_)) / 6.0),
          static_cast<float>((state.correct + cfg.user_alpha * concept_acc) /
                             (state.count + cfg.user_alpha)),
          static_cast<float>(std::log1p(static_cast<double>(state.count)) / 5.0),
          static_cast<float>(window_.Mean(mean)),
          static_cast<float>(std::tanh(static_cast<double>(streak_) / 5.0)),
          static_cast<float>(gap)};
}

void StudentState::Update(const Interaction& row) {
  auto& state = concepts_[row.concept_id];
  ++state.count;
  state.correct += row.correct;
  state.last_step = count_;
  state.last_timestamp = row.timestamp_sec;
  ++count_;
  correct_ += row.correct;
  streak_ = row.correct == 1 ? std::max<std::int64_t>(streak_, 0) + 1
                             : std::min<std::int64_t>(streak_, 0) - 1;
  window_.Push(row.correct);
}

HistoricalStatEngine::HistoricalStatEngine(StatConfig config) : config_(config) {
  if (config.window_size == 0 || config.window_size > kMaxWindowSize ||
      !std::isfinite(config.prior_alpha) || config.prior_alpha <= 0.0 ||
      !std::isfinite(config.user_alpha) || config.user_alpha <= 0.0) {
    throw std::invalid_argument("invalid statistics configuration");
  }
}

void HistoricalStatEngine::FitFlat(const Interaction* rows, std::size_t num_rows,
                                  const std::int64_t* offsets, std::size_t num_seqs) {
  ValidateFlat(rows, num_rows, offsets, num_seqs);
  if (num_rows == 0) { throw std::invalid_argument("empty training data"); }
  const auto limits = DenseLimits(rows, num_rows);
  Aggregate next(limits.first, limits.second);
  for (std::size_t i = 0; i < num_rows; ++i) { next.Add(rows[i]); }
  totals_ = std::move(next);
  fitted_ = true;
}

template <bool ExcludeFold>
void HistoricalStatEngine::TransformSequence(
    const Interaction* rows, std::size_t begin, std::size_t end,
    const Aggregate* excluded, float* output) const {
  auto global = totals_.global;
  if constexpr (ExcludeFold) { global = global.Minus(excluded->global); }
  const double mean = static_cast<double>(global.correct) / global.count;
  const double mean_rt = global.response_sum / global.count;
  StudentState state(config_.window_size);
  for (std::size_t i = begin; i < end; ++i) {
    const auto& row = rows[i];
    auto item = totals_.items.Get(row.item_id);
    auto concept_count = totals_.concepts.Get(row.concept_id);
    if constexpr (ExcludeFold) {
      item = item.Minus(excluded->items.Get(row.item_id));
      concept_count = concept_count.Minus(excluded->concepts.Get(row.concept_id));
    }
    const double a = config_.prior_alpha;
    const double item_acc = (item.correct + a * mean) / (item.count + a);
    const double concept_acc = (concept_count.correct + a * mean) / (concept_count.count + a);
    const double item_rt = (std::max(0.0, item.response_sum) + a * mean_rt) / (item.count + a);
    const auto feature = state.Extract(row, config_, mean, item_acc, concept_acc, item_rt);
    std::copy(feature.begin(), feature.end(), output + i * kNumStatFeatures);
    state.Update(row);
  }
}

void HistoricalStatEngine::TransformFlat(
    const Interaction* rows, std::size_t num_rows, const std::int64_t* offsets,
    std::size_t num_seqs, std::size_t num_threads, float* output,
    std::size_t output_size) const {
  ValidateFlat(rows, num_rows, offsets, num_seqs);
  ValidateOutput(num_rows, num_threads, output, output_size);
  if (!fitted_) { throw std::invalid_argument("engine not fitted"); }
  ParallelSequences(num_seqs, num_threads, [&](std::size_t s) {
    TransformSequence<false>(rows, static_cast<std::size_t>(offsets[s]),
                      static_cast<std::size_t>(offsets[s + 1]), nullptr, output);
  });
}

void HistoricalStatEngine::FitAndTransformOOF(
    const Interaction* rows, std::size_t num_rows, const std::int64_t* offsets,
    std::size_t num_seqs, const std::int32_t* fold_ids, std::size_t num_folds,
    std::size_t num_threads, float* output, std::size_t output_size) {
  ValidateFlat(rows, num_rows, offsets, num_seqs);
  ValidateOutput(num_rows, num_threads, output, output_size);
  if (fold_ids == nullptr || num_folds < 2 || num_folds > num_seqs || num_rows == 0) {
    throw std::invalid_argument("invalid OOF configuration");
  }
  const auto limits = DenseLimits(rows, num_rows);
  Aggregate next(limits.first, limits.second);
  std::vector<Aggregate> folds;
  folds.reserve(num_folds);
  for (std::size_t fold = 0; fold < num_folds; ++fold) {
    folds.emplace_back(limits.first, limits.second);
  }
  std::unordered_map<std::int32_t, std::int32_t> student_folds;
  for (std::size_t s = 0; s < num_seqs; ++s) {
    const auto fold = fold_ids[s];
    if (fold < 0 || static_cast<std::size_t>(fold) >= num_folds) {
      throw std::invalid_argument("fold ID outside range");
    }
    if (offsets[s] < offsets[s + 1]) {
      const auto result = student_folds.emplace(rows[offsets[s]].user_id, fold);
      if (!result.second && result.first->second != fold) {
        throw std::invalid_argument("student spans OOF folds");
      }
    }
    for (auto i = offsets[s]; i < offsets[s + 1]; ++i) {
      next.Add(rows[i]);
      folds[static_cast<std::size_t>(fold)].Add(rows[i]);
    }
  }
  for (const auto& fold : folds) {
    if (fold.global.count == 0 || fold.global.count == next.global.count) {
      throw std::invalid_argument("empty OOF fold or complement");
    }
  }
  totals_ = std::move(next);
  fitted_ = true;
  ParallelSequences(num_seqs, num_threads, [&](std::size_t s) {
    TransformSequence<true>(rows, static_cast<std::size_t>(offsets[s]),
                      static_cast<std::size_t>(offsets[s + 1]),
                      &folds[static_cast<std::size_t>(fold_ids[s])], output);
  });
}
}  // namespace hsf
