#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct ggml_cuda_kv_stream_share_key {
    uint32_t width;   // Q->ne[1]
    uint32_t bucket;  // floor(log2(streamed pages))
};

// Chooses the CPU share per kind of decode graph from decode graph timings
class ggml_cuda_kv_stream_share_tuner {
public:
    static constexpr uint32_t N_ARMS = 4;
    static constexpr float SHARES[N_ARMS] = { 0.0f, 0.2f, 0.4f, 0.6f };

    explicit ggml_cuda_kv_stream_share_tuner(
            uint32_t kept_samples = 8,
            double margin = 0.02,
            uint32_t warmup_samples = 1) :
        kept_samples_(std::max<uint32_t>(kept_samples, 1)),
        margin_(std::isfinite(margin) ? std::clamp(margin, 0.0, 0.999) : 0.02),
        warmup_samples_(warmup_samples) {
    }

    static uint32_t bucket(uint32_t streamed_pages) {
        uint32_t result = 0;
        while (streamed_pages >>= 1) {
            ++result;
        }
        return result;
    }

    uint32_t arm(const ggml_cuda_kv_stream_share_key & key) {
        entry & e = table_[pack(key)];
        if (e.decided) {
            return e.verdict;
        }
        for (uint32_t i = 0; i < N_ARMS; ++i) {
            const uint32_t candidate = (e.next + i) % N_ARMS;
            if (e.kept[candidate].size() < kept_samples_) {
                e.next = (candidate + 1) % N_ARMS;
                return candidate;
            }
        }
        return 0;
    }

    void observe(const ggml_cuda_kv_stream_share_key & key, uint32_t arm, double elapsed_ms) {
        if (arm >= N_ARMS || !std::isfinite(elapsed_ms) || elapsed_ms <= 0.0) {
            return;
        }
        entry & e = table_[pack(key)];
        if (e.decided || e.kept[arm].size() >= kept_samples_) {
            return;
        }
        if (e.warmups[arm] < warmup_samples_) {
            ++e.warmups[arm];
            return;
        }
        e.kept[arm].push_back(elapsed_ms);
        for (const std::vector<double> & samples : e.kept) {
            if (samples.size() < kept_samples_) {
                return;
            }
        }

        std::array<double, N_ARMS> medians;
        for (uint32_t i = 0; i < N_ARMS; ++i) {
            medians[i] = median(e.kept[i]);
        }
        const double limit = (1.0 - margin_) * medians[0];
        double best = INFINITY;
        for (uint32_t i = 1; i < N_ARMS; ++i) {
            if (medians[i] <= limit) {
                best = std::min(best, medians[i]);
            }
        }
        for (uint32_t i = 1; i < N_ARMS; ++i) {
            if (medians[i] <= limit && medians[i] <= (1.0 + margin_) * best) {
                e.verdict = i;
                break;
            }
        }
        e.decided = true;
    }

    bool decided(const ggml_cuda_kv_stream_share_key & key) const {
        const auto it = table_.find(pack(key));
        return it != table_.end() && it->second.decided;
    }

    double median_ms(const ggml_cuda_kv_stream_share_key & key, uint32_t arm) const {
        const auto it = table_.find(pack(key));
        return it == table_.end() || arm >= N_ARMS ? 0.0 : median(it->second.kept[arm]);
    }

    void reset() { table_.clear(); }

private:
    struct entry {
        std::array<std::vector<double>, N_ARMS> kept;
        std::array<uint32_t, N_ARMS> warmups = {};
        uint32_t next = 0;
        uint32_t verdict = 0;
        bool decided = false;
    };

    static uint64_t pack(const ggml_cuda_kv_stream_share_key & key) {
        return (uint64_t(key.width) << 32) | key.bucket;
    }

    static double median(std::vector<double> samples) {
        if (samples.empty()) {
            return 0.0;
        }
        std::sort(samples.begin(), samples.end());
        const size_t mid = samples.size() / 2;
        return samples.size() % 2 ? samples[mid] : 0.5 * (samples[mid - 1] + samples[mid]);
    }

    uint32_t kept_samples_ = 8;
    double margin_ = 0.02;
    uint32_t warmup_samples_ = 1;
    std::unordered_map<uint64_t, entry> table_;
};
