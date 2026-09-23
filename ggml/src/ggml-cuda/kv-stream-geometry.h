#pragma once

#include <cstdint>

constexpr int GGML_CUDA_KV_STREAM_HEAD_DIM = 256;
constexpr int GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS = 32;
constexpr uint32_t GGML_CUDA_KV_STREAM_PAGE_TOKENS = 256;
