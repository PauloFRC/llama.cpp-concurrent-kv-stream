#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// Chooses between greedy and fixed copy batches using decode graph timings.
class ggml_cuda_kv_stream_span_tuner {
public:
    explicit ggml_cuda_kv_stream_span_tuner(
            uint32_t trial_samples = 16,
            double minimum_relative_gain = 0.005,
            uint32_t warmup_samples = 1) :
        trial_samples_(std::max<uint32_t>(trial_samples, 1)),
        minimum_relative_gain_(
                std::isfinite(minimum_relative_gain) ?
                    std::clamp(minimum_relative_gain, 0.0, 0.999) : 0.005),
        warmup_samples_(warmup_samples) {
    }

    void reset() {
        greedy_ = false;
        selected_ = false;
        fixed_samples_ = 0;
        greedy_samples_ = 0;
        fixed_warmups_ = 0;
        greedy_warmups_ = 0;
        fixed_ms_ = 0.0;
        greedy_ms_ = 0.0;
    }

    void observe(double elapsed_ms, bool streamed, bool sample_was_greedy) {
        if (selected_ || !streamed || !std::isfinite(elapsed_ms) || elapsed_ms <= 0.0) {
            return;
        }

        uint32_t & warmups = sample_was_greedy ? greedy_warmups_ : fixed_warmups_;
        if (warmups < warmup_samples_) {
            ++warmups;
            return;
        }

        if (sample_was_greedy) {
            greedy_ms_ += elapsed_ms;
            ++greedy_samples_;
        } else {
            fixed_ms_ += elapsed_ms;
            ++fixed_samples_;
        }

        if (fixed_samples_ < trial_samples_) {
            greedy_ = false;
            return;
        }

        if (greedy_samples_ < trial_samples_) {
            greedy_ = true;
            return;
        }

        const double fixed_average = fixed_ms_ / fixed_samples_;
        const double greedy_average = greedy_ms_ / greedy_samples_;
        greedy_ = greedy_average < fixed_average * (1.0 - minimum_relative_gain_);
        selected_ = true;
    }

    bool use_greedy_batch() const { return greedy_; }
    bool selected() const { return selected_; }
    double fixed_average_ms() const {
        return fixed_samples_ == 0 ? 0.0 : fixed_ms_ / fixed_samples_;
    }
    double greedy_average_ms() const {
        return greedy_samples_ == 0 ? 0.0 : greedy_ms_ / greedy_samples_;
    }

private:
    uint32_t trial_samples_ = 16;
    double minimum_relative_gain_ = 0.005;
    uint32_t warmup_samples_ = 1;
    uint32_t fixed_warmups_ = 0;
    uint32_t greedy_warmups_ = 0;
    bool greedy_ = false;
    bool selected_ = false;
    uint32_t fixed_samples_ = 0;
    uint32_t greedy_samples_ = 0;
    double fixed_ms_ = 0.0;
    double greedy_ms_ = 0.0;
};
