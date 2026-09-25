#pragma once

#include "ggml.h"
#include "kv-stream-geometry.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// why a layer cannot hand its pages to the CPU; BLOCK_OK means it can
enum ggml_cuda_kv_stream_cpu_split_block {
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_OK = 0,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_K_TYPE,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_V_TYPE,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_Q_TYPE,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_Q_GAP,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_HEAD_DIM,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_QUERY_TOKENS,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_SEQ_DIM,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MASK,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MASK_ROWS,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MAX_BIAS,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_SOFT_CAP,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MASK_CHANGED,
    GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_COUNT,
};

inline ggml_cuda_kv_stream_cpu_split_block ggml_cuda_kv_stream_cpu_split_get_block(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (K->type != GGML_TYPE_Q8_0) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_K_TYPE;
    }
    if (V->type != GGML_TYPE_Q4_0) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_V_TYPE;
    }
    if (Q->type != GGML_TYPE_F32) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_Q_TYPE;
    }
    if (ggml_nbytes(Q) != size_t(ggml_nelements(Q))*sizeof(float)) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_Q_GAP;
    }
    if (Q->ne[0] != GGML_CUDA_KV_STREAM_HEAD_DIM || V->ne[0] != GGML_CUDA_KV_STREAM_HEAD_DIM) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_HEAD_DIM;
    }
    if (Q->ne[1] > GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_QUERY_TOKENS;
    }
    if (Q->ne[3] != 1) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_SEQ_DIM;
    }
    if (mask == nullptr) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MASK;
    }
    if (mask->ne[2] != 1) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MASK_ROWS;
    }
    if (max_bias != 0.0f) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MAX_BIAS;
    }
    if (logit_softcap != 0.0f) {
        return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_SOFT_CAP;
    }
    return GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_OK;
}

inline const char * ggml_cuda_kv_stream_cpu_split_block_reason(ggml_cuda_kv_stream_cpu_split_block block) {
    switch (block) {
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_OK:           return nullptr;
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_K_TYPE:       return "K cache is not q8_0";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_V_TYPE:       return "V cache is not q4_0";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_Q_TYPE:       return "Q is not f32";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_Q_GAP:        return "Q has gaps between tokens";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_HEAD_DIM:     return "head dim is not 256";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_QUERY_TOKENS: return "the decode batch is wider than 32 tokens";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_SEQ_DIM:      return "the batch has more than one sequence dimension";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MASK:         return "the layer has no mask";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MASK_ROWS:    return "the mask has more than one row per token";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MAX_BIAS:     return "attention has a max_bias";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_SOFT_CAP:     return "attention has a logit softcap";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MASK_CHANGED: return "another mask already owns this graph";
        case GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_COUNT:        break;
    }
    return "unknown";
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
