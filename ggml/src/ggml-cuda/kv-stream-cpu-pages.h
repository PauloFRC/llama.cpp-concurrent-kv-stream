#pragma once

#include "ggml.h"
#include "kv-stream-geometry.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

inline bool ggml_cuda_kv_stream_cpu_split_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    return K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0 &&
        Q->type == GGML_TYPE_F32 && ggml_nbytes(Q) == size_t(ggml_nelements(Q))*sizeof(float) &&
        Q->ne[0] == GGML_CUDA_KV_STREAM_HEAD_DIM && V->ne[0] == GGML_CUDA_KV_STREAM_HEAD_DIM &&
        Q->ne[1] <= GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS && Q->ne[3] == 1 &&
        mask != nullptr && mask->ne[2] == 1 &&
        max_bias == 0.0f && logit_softcap == 0.0f;
}

struct ggml_cuda_kv_stream_page_state {
    uint32_t n_pages;
    uint32_t resident_pages;
    const std::vector<uint8_t> & live_pages; // empty: every page is live
    const std::vector<uint32_t> & mutable_pages;
    bool all_pages_mutable;

    ggml_cuda_kv_stream_page_state(
            uint32_t n_pages, uint32_t resident_pages,
            const std::vector<uint8_t> & live_pages,
            const std::vector<uint32_t> & mutable_pages, bool all_pages_mutable) :
        n_pages(n_pages), resident_pages(resident_pages), live_pages(live_pages),
        mutable_pages(mutable_pages), all_pages_mutable(all_pages_mutable) {
        GGML_ASSERT(live_pages.empty() || live_pages.size() == n_pages);
        GGML_ASSERT(std::is_sorted(mutable_pages.begin(), mutable_pages.end()));
    }

    bool live(uint32_t page) const {
        GGML_ASSERT(page < n_pages);
        return live_pages.empty() || live_pages[page] != 0;
    }

    bool immutable(uint32_t page) const {
        GGML_ASSERT(page < n_pages);
        return page + 1 < n_pages && !all_pages_mutable &&
            !std::binary_search(mutable_pages.begin(), mutable_pages.end(), page);
    }
};

inline std::vector<uint32_t> ggml_cuda_kv_stream_select_cpu_pages(
        const ggml_cuda_kv_stream_page_state & state, uint32_t n_cpu) {
    std::vector<uint32_t> pages;
    pages.reserve(std::min(n_cpu, state.n_pages));
    for (uint32_t page = state.n_pages; page > state.resident_pages && pages.size() < n_cpu; --page) {
        if (state.live(page - 1) && state.immutable(page - 1)) {
            pages.push_back(page - 1);
        }
    }
    std::reverse(pages.begin(), pages.end());
    return pages;
}
