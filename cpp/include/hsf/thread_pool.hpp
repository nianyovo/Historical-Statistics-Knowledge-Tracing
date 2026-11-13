#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace hsf {
class ThreadPool {
 public:
  explicit ThreadPool(std::size_t num_threads) {
    if (num_threads == 0) { throw std::invalid_argument("zero threads"); }
    try {
      for (std::size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] { Run(); });
      }
    } catch (...) {
      Stop();
      throw;
    }
  }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;
  ~ThreadPool() { Stop(); }

  std::future<void> Submit(std::function<void()> function) {
    auto task = std::make_shared<std::packaged_task<void()>>(std::move(function));
    auto future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) { throw std::runtime_error("stopped pool"); }
      tasks_.emplace([task] { (*task)(); });
    }
    ready_.notify_one();
    return future;
  }

 private:
  void Run() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty()) { return; }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }
  void Stop() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) { worker.join(); }
    }
  }
  std::mutex mutex_;
  std::condition_variable ready_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
  bool stopping_ = false;
};
}  // namespace hsf
