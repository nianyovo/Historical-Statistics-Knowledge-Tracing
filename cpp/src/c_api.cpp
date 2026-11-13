#include "hsf/c_api.h"

#include <memory>
#include <new>
#include <stdexcept>
#include "hsf/stat_engine.hpp"

namespace {
template <typename Function>
int Guard(Function function) noexcept {
  try {
    function();
    return static_cast<int>(hsf::StatusCode::kOk);
  } catch (const std::invalid_argument&) {
    return static_cast<int>(hsf::StatusCode::kInvalidInput);
  } catch (const std::bad_alloc&) {
    return static_cast<int>(hsf::StatusCode::kAllocation);
  } catch (...) {
    return static_cast<int>(hsf::StatusCode::kInternal);
  }
}
hsf::HistoricalStatEngine& Engine(void* handle) {
  if (handle == nullptr) { throw std::invalid_argument("null handle"); }
  return *static_cast<hsf::HistoricalStatEngine*>(handle);
}
}  // namespace

extern "C" {
int hsf_create_engine(const HsfConfig* config, void** out_handle) {
  return Guard([&] {
    if (out_handle == nullptr) { throw std::invalid_argument("null handle output"); }
    *out_handle = nullptr;
    hsf::StatConfig cfg;
    if (config != nullptr) {
      if ((config->has_real_timestamp != 0 && config->has_real_timestamp != 1) ||
          (config->has_real_resp_time != 0 && config->has_real_resp_time != 1)) {
        throw std::invalid_argument("flags must be boolean");
      }
      cfg = {config->window_size, config->prior_alpha, config->user_alpha,
             config->has_real_timestamp != 0, config->has_real_resp_time != 0};
    }
    auto engine = std::make_unique<hsf::HistoricalStatEngine>(cfg);
    *out_handle = engine.release();
  });
}
int hsf_destroy_engine(void* handle) {
  return Guard([&] {
    std::unique_ptr<hsf::HistoricalStatEngine> owner(
        static_cast<hsf::HistoricalStatEngine*>(handle));
  });
}
int hsf_fit_flat(void* handle, const HsfInteraction* rows, size_t num_rows,
                 const int64_t* offsets, size_t num_seqs) {
  return Guard([&] { Engine(handle).FitFlat(rows, num_rows, offsets, num_seqs); });
}
int hsf_transform_flat(void* handle, const HsfInteraction* rows, size_t num_rows,
                       const int64_t* offsets, size_t num_seqs, size_t num_threads,
                       float* output, size_t output_size) {
  return Guard([&] { Engine(handle).TransformFlat(rows, num_rows, offsets,
                        num_seqs, num_threads, output, output_size); });
}
int hsf_fit_transform_oof_flat(
    void* handle, const HsfInteraction* rows, size_t num_rows,
    const int64_t* offsets, size_t num_seqs, const int32_t* fold_ids,
    size_t num_folds, size_t num_threads, float* output, size_t output_size) {
  return Guard([&] { Engine(handle).FitAndTransformOOF(rows, num_rows, offsets,
                        num_seqs, fold_ids, num_folds, num_threads, output, output_size); });
}
}
