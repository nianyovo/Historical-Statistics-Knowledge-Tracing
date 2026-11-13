#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include "hsf/types.hpp"

namespace hsf {
// Fixed stack storage; Push never allocates. Runtime window is configurable.
template <typename T, std::size_t Capacity = kMaxWindowSize>
class RingBuffer {
  static_assert(std::is_arithmetic<T>::value, "RingBuffer requires numbers");
 public:
  explicit RingBuffer(std::size_t capacity = 5) : capacity_(capacity) {
    if (capacity == 0 || capacity > Capacity) {
      throw std::invalid_argument("window_size out of range");
    }
  }
  void Push(T value) noexcept {
    if (size_ == capacity_) {
      sum_ -= static_cast<double>(values_[head_]);
    } else {
      ++size_;
    }
    values_[head_] = value;
    sum_ += static_cast<double>(value);
    head_ = (head_ + 1) % capacity_;
  }
  double Mean(double empty_value) const noexcept {
    return size_ == 0 ? empty_value : sum_ / static_cast<double>(size_);
  }
  std::size_t Size() const noexcept { return size_; }

 private:
  std::array<T, Capacity> values_{};
  std::size_t capacity_;
  std::size_t size_ = 0;
  std::size_t head_ = 0;
  double sum_ = 0.0;
};
}  // namespace hsf
