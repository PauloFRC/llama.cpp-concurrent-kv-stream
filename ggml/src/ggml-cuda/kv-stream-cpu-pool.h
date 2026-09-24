#pragma once

#include "ggml.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

// jobs run one at a time in arm order, each on every thread; job ids wrap and compare cyclically
template <typename job_t>
class ggml_cuda_kv_stream_cpu_pool {
public:
    using body_fn = std::function<void(const job_t & job, int thread, int n_threads)>;

    ggml_cuda_kv_stream_cpu_pool(int n_threads, uint32_t depth, body_fn body, std::atomic<uint32_t> * done = nullptr) :
            n_threads_(n_threads), jobs_(depth), body_(std::move(body)), done_(done), pending_(n_threads) {
        GGML_ASSERT(n_threads > 0 && depth > 0);
        threads_.reserve(n_threads);
        for (int thread = 0; thread < n_threads; ++thread) {
            threads_.emplace_back([this, thread] { run(thread); });
        }
    }

    ~ggml_cuda_kv_stream_cpu_pool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            GGML_ASSERT(released_ == armed_);
            done_cv_.wait(lock, [this] { return completed_ == armed_; });
            stop_ = true;
        }
        ready_cv_.notify_all();
        for (auto & thread : threads_) {
            thread.join();
        }
    }

    // never blocks: the owner sizes the queue for every job it arms before draining
    uint32_t arm(job_t job) {
        std::lock_guard<std::mutex> lock(mutex_);
        GGML_ASSERT(armed_ - completed_ < jobs_.size());
        ++armed_;
        jobs_[armed_ % jobs_.size()] = std::move(job);
        return uint32_t(armed_);
    }

    // called from a CUDA host function, so it makes no CUDA calls
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            GGML_ASSERT(released_ != armed_);
            ++released_;
        }
        ready_cv_.notify_all();
    }

    void wait(uint32_t job) {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&] { return int32_t(uint32_t(completed_) - job) >= 0; });
    }

    void wait_idle() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return completed_ == armed_; });
    }

    bool idle() {
        std::lock_guard<std::mutex> lock(mutex_);
        return completed_ == armed_;
    }

private:
    void run(int thread) {
        uint64_t last = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_cv_.wait(lock, [&] { return stop_ || (released_ != completed_ && last == completed_); });
            if (stop_) {
                return;
            }
            const uint64_t job = completed_ + 1;
            lock.unlock();
            body_(jobs_[job % jobs_.size()], thread, n_threads_);
            lock.lock();
            last = job;
            if (--pending_ == 0) {
                pending_ = n_threads_;
                completed_ = job;
                if (done_ != nullptr) {
                    done_->store(uint32_t(job), std::memory_order_release);
                }
                ready_cv_.notify_all();
                done_cv_.notify_all();
            }
        }
    }

    const int n_threads_;
    std::vector<job_t> jobs_;
    body_fn body_;
    std::atomic<uint32_t> * done_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable ready_cv_;
    std::condition_variable done_cv_;
    uint64_t armed_ = 0;
    uint64_t released_ = 0;
    uint64_t completed_ = 0;
    int pending_;
    bool stop_ = false;
};
