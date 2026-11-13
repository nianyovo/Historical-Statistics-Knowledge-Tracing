#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include "hsf/ring_buffer.hpp"
#include "hsf/types.hpp"

namespace hsf {
class CounterTable {
 public:
  explicit CounterTable(std::size_t dense_limit = 0) : dense_(dense_limit + 1) {}
  void Add(std::int32_t key, const Interaction& row) {
    if (key > 0 && static_cast<std::size_t>(key) < dense_.size()) {
      dense_[static_cast<std::size_t>(key)].Add(row);
    } else {
      sparse_[key].Add(row);
    }
  }
  Counter Get(std::int32_t key) const {
    if (key > 0 && static_cast<std::size_t>(key) < dense_.size()) {
      return dense_[static_cast<std::size_t>(key)];
    }
    const auto found = sparse_.find(key);
    return found == sparse_.end() ? Counter{} : found->second;
  }

 private:
  std::vector<Counter> dense_;
  std::unordered_map<std::int32_t, Counter> sparse_;
};

struct Aggregate {
  explicit Aggregate(std::size_t item_limit = 0, std::size_t concept_limit = 0)
      : items(item_limit), concepts(concept_limit) {}
  Counter global;
  CounterTable items;
  CounterTable concepts;
  void Add(const Interaction& row);
};

struct ConceptState {
  std::int64_t count = 0;
  std::int64_t correct = 0;
  std::int64_t last_step = 0;
  double last_timestamp = 0.0;
};

class StudentState {
 public:
  explicit StudentState(std::size_t window_size) : window_(window_size) {}
  std::array<float, kNumStatFeatures> Extract(
      const Interaction& row, const StatConfig& config, double mean,
      double item_acc, double concept_acc, double item_rt) const;
  void Update(const Interaction& row);

 private:
  std::int64_t count_ = 0;
  std::int64_t correct_ = 0;
  std::int64_t streak_ = 0;
  RingBuffer<int> window_;
  std::unordered_map<std::int32_t, ConceptState> concepts_;
};

class HistoricalStatEngine {
 public:
  explicit HistoricalStatEngine(StatConfig config = {});
  void FitFlat(const Interaction* rows, std::size_t num_rows,
               const std::int64_t* offsets, std::size_t num_seqs);
  void TransformFlat(const Interaction* rows, std::size_t num_rows,
                     const std::int64_t* offsets, std::size_t num_seqs,
                     std::size_t num_threads, float* output,
                     std::size_t output_size) const;
  void FitAndTransformOOF(const Interaction* rows, std::size_t num_rows,
                          const std::int64_t* offsets, std::size_t num_seqs,
                          const std::int32_t* fold_ids, std::size_t num_folds,
                          std::size_t num_threads, float* output,
                          std::size_t output_size);

 private:
  template <bool ExcludeFold>
  void TransformSequence(const Interaction* rows, std::size_t begin,
                         std::size_t end, const Aggregate* excluded,
                         float* output) const;
  StatConfig config_;
  Aggregate totals_;
  bool fitted_ = false;
};
}  // namespace hsf
