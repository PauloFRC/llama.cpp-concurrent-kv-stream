#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

struct ggml_cuda_kv_stream_cpu_attn_params {
    const uint8_t * k = nullptr;
    size_t k_token_stride = 0;
    size_t k_head_stride = 0;
    const uint8_t * v = nullptr;
    size_t v_token_stride = 0;
    size_t v_head_stride = 0;
    const float * q = nullptr;
    size_t q_token_stride = 0;
    size_t q_head_stride = 0;
    const uint16_t * mask = nullptr;
    size_t mask_token_stride = 0;
    const uint32_t * pages = nullptr;
    uint32_t n_pages = 0;
    uint32_t page_tokens = 0;   // multiple of 16
    int n_head = 0;
    int n_head_kv = 0;
    int n_tokens = 0;           // at most GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS
    float scale = 0.0f;
    float * out = nullptr;      // row token*n_head + head, head dim floats, not normalized
    float * out_meta = nullptr;
};

// only _supported() is safe to call anywhere, the rest abort or trap without AVX-512
GGML_BACKEND_API bool ggml_cuda_kv_stream_cpu_attn_supported();
GGML_BACKEND_API void ggml_cuda_kv_stream_cpu_attn_init(float * out, float * out_meta, int nrows);
GGML_BACKEND_API void ggml_cuda_kv_stream_cpu_attn(const ggml_cuda_kv_stream_cpu_attn_params & params);
GGML_BACKEND_API void ggml_cuda_kv_stream_cpu_attn_fold(
    float * acc, float * acc_meta, const float * part, const float * part_meta, int nrows);
