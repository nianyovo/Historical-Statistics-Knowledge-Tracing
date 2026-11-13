#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "hsf/stat_engine.hpp"

namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) { throw std::runtime_error(message); }
}
struct Batch {
  std::vector<hsf::Interaction> rows;
  std::vector<std::int64_t> offsets{0};
  std::vector<std::int32_t> folds;
};
Batch MakeBatch(std::size_t students, std::size_t length) {
  Batch batch;
  std::mt19937 rng(42);
  for (std::size_t s = 0; s < students; ++s) {
    for (std::size_t t = 0; t < length; ++t) {
      batch.rows.push_back({static_cast<std::int32_t>(s),
          static_cast<std::int32_t>(1 + (s * length + t) % 5000),
          static_cast<std::int32_t>(1 + t % 20), static_cast<std::int32_t>(rng() % 2),
          static_cast<double>(t * 100), 20.0 + static_cast<double>(rng() % 100)});
    }
    batch.offsets.push_back(static_cast<std::int64_t>(batch.rows.size()));
    batch.folds.push_back(static_cast<std::int32_t>(s % 5));
  }
  return batch;
}
std::vector<float> Oof(hsf::HistoricalStatEngine& engine, const Batch& batch,
                       std::size_t threads) {
  std::vector<float> out(batch.rows.size() * hsf::kNumStatFeatures);
  engine.FitAndTransformOOF(batch.rows.data(), batch.rows.size(), batch.offsets.data(),
      batch.folds.size(), batch.folds.data(), 5, threads, out.data(), out.size());
  return out;
}
std::vector<float> Transform(hsf::HistoricalStatEngine& engine, const Batch& batch) {
  std::vector<float> out(batch.rows.size() * hsf::kNumStatFeatures);
  engine.TransformFlat(batch.rows.data(), batch.rows.size(), batch.offsets.data(),
      batch.folds.size(), 4, out.data(), out.size());
  return out;
}
void Test() {
  hsf::RingBuffer<int> ring(2);
  Require(ring.Mean(0.25) == 0.25, "empty ring");
  ring.Push(1); ring.Push(0); ring.Push(0);
  Require(ring.Size() == 2 && ring.Mean(1) == 0, "ring eviction");
  auto batch = MakeBatch(10, 12);
  hsf::HistoricalStatEngine engine;
  const auto before = Oof(engine, batch, 1);
  Require(before == Oof(engine, batch, 4), "parallel determinism");
  auto changed = batch;
  for (std::size_t i = 0; i < 12; ++i) { changed.rows[i].correct ^= 1; }
  const auto after = Oof(engine, changed, 4);
  for (std::size_t i = 0; i < 12; ++i) {
    Require(before[i * 10] == after[i * 10] && before[i * 10 + 1] == after[i * 10 + 1],
            "OOF self leakage");
  }
  engine.FitFlat(batch.rows.data(), batch.rows.size(), batch.offsets.data(), batch.folds.size());
  const auto causal_before = Transform(engine, batch);
  changed = batch;
  for (std::size_t i = 5; i < 12; ++i) {
    changed.rows[i].correct ^= 1;
    changed.rows[i].resp_time_sec += 100;
  }
  const auto causal_after = Transform(engine, changed);
  for (std::size_t i = 0; i < 60; ++i) {
    Require(causal_before[i] == causal_after[i], "within-step leakage");
  }
  changed.folds[1] = 0;
  changed.rows[12].user_id = 0;
  bool rejected = false;
  try { Oof(engine, changed, 1); } catch (const std::invalid_argument&) { rejected = true; }
  Require(rejected, "invalid mixed-student sequence accepted");
  std::cout << "Native rigor tests passed\n";
}
void Benchmark() {
  const auto batch = MakeBatch(5000, 200);
  hsf::HistoricalStatEngine engine;
  for (std::size_t threads : {1U, 2U, 4U, 8U}) {
    const auto start = std::chrono::steady_clock::now();
    const auto features = Oof(engine, batch, threads);
    const auto ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "interactions=" << batch.rows.size() << " threads=" << threads
              << " oof_ms=" << ms << " checksum=" << features.front() << '\n';
  }
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Test();
    if (argc < 2 || std::string(argv[1]) != "--test-only") { Benchmark(); }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
