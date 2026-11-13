#pragma once

#include <cstddef>
#include <cstdint>
#include "hsf/c_api.h"

namespace hsf {
constexpr std::size_t kNumStatFeatures = 10;
constexpr std::size_t kMaxWindowSize = 1024;
enum class StatusCode : int { kOk = 0, kInvalidInput = -1, kAllocation = -2, kInternal = -3 };
using Interaction = HsfInteraction;
static_assert(sizeof(Interaction) == 32, "Unexpected interaction ABI layout");

struct StatConfig {
  std::size_t window_size = 5;
  double prior_alpha = 10.0;
  double user_alpha = 2.0;
  bool has_real_timestamp = true;
  bool has_real_resp_time = true;
};

struct Counter {
  std::int64_t count = 0;
  std::int64_t correct = 0;
  double response_sum = 0.0;
  void Add(const Interaction& row) noexcept {
    ++count;
    correct += row.correct;
    response_sum += row.resp_time_sec;
  }
  Counter Minus(const Counter& other) const noexcept {
    return {count - other.count, correct - other.correct,
            response_sum - other.response_sum};
  }
};
}  // namespace hsf
