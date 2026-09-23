#include "kv-stream-cpu-attn.h"
#include "kv-stream-geometry.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(GGML_CUDA_KV_STREAM_CPU_ATTN_AVX512)

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <vector>

namespace {

constexpr int D          = GGML_CUDA_KV_STREAM_HEAD_DIM;
constexpr int MAX_TOKENS = GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS;
constexpr int NB         = D/QK8_0;
constexpr int NP         = NB/2;   // one zmm holds two blocks
constexpr int TILE       = 16;

static_assert(QK8_0 == QK4_0, "K and V blocks must have the same length");
static_assert(D%(2*QK8_0) == 0, "head dim must fill whole zmm block pairs");
static_assert(TILE*sizeof(uint16_t) == sizeof(__m256i), "a mask tile must fill one ymm");

inline float f16_to_f32(ggml_half h) { return _cvtsh_ss(h); }

inline __m512 v_expf(__m512 x) {
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                         _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __mmask16 d = _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b, _mm512_set1_ps(0x1.573e2ep-5f)), u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b, _mm512_set1_ps(0x1.fffdb6p-2f))),
        u, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
    const __m512 res = _mm512_scalef_ps(j, n);
    if (_mm512_kortestz(d, d)) { return res; }
    const __m512 zero = _mm512_setzero_ps();
    const __m512 alt = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
    return _mm512_mask_blend_ps(d, res, alt);
}

// lanes 0-7 = a, lanes 8-15 = b, two blocks per register
inline __m512 lane_pair(float a, float b) {
    return _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(a)), _mm256_set1_ps(b), 1);
}

struct k_cell {
    __m512i qs[NP];
    __m512  dk[NP];
    __m512  bias[NP];  // 128*sum(qs)*dk, cancels the u8 offset on Q
};

inline void prepare_k(const block_q8_0 * blocks, k_cell & kc) {
    const __m512i ones = _mm512_set1_epi8(1);
    for (int j = 0; j < NP; ++j) {
        const __m256i lo = _mm256_loadu_si256((const __m256i *) blocks[2*j].qs);
        const __m256i hi = _mm256_loadu_si256((const __m256i *) blocks[2*j + 1].qs);
        kc.qs[j] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        kc.dk[j] = lane_pair(f16_to_f32(blocks[2*j].d), f16_to_f32(blocks[2*j + 1].d));
        const __m512i ksum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, kc.qs[j]);
        kc.bias[j] = _mm512_mul_ps(_mm512_cvtepi32_ps(ksum), _mm512_mul_ps(kc.dk[j], _mm512_set1_ps(128.0f)));
    }
}

// Q row as u8 (q + 128) for the unsigned vpdpbusd operand
struct q_vec {
    __m512i qu[NP];
    __m512  dq[NP];
};

inline void quantize_q(const float * x, q_vec & qv) {
    alignas(64) int8_t qs[D];
    float d[NB];
    for (int b = 0; b < NB; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < QK8_0; ++i) { amax = std::max(amax, std::fabs(x[b*QK8_0 + i])); }
        d[b] = amax/127.0f;
        const float id = d[b] != 0.0f ? 1.0f/d[b] : 0.0f;
        for (int i = 0; i < QK8_0; ++i) { qs[b*QK8_0 + i] = int8_t(std::lrintf(x[b*QK8_0 + i]*id)); }
    }
    const __m512i off = _mm512_set1_epi8(int8_t(0x80));
    for (int j = 0; j < NP; ++j) {
        qv.qu[j] = _mm512_xor_si512(_mm512_load_si512((const __m512i *)(qs + 2*QK8_0*j)), off);
        qv.dq[j] = lane_pair(d[2*j], d[2*j + 1]);
    }
}

inline float score(const q_vec & qv, const k_cell & kc) {
    __m512 acc = _mm512_setzero_ps();
    for (int j = 0; j < NP; ++j) {
        const __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qv.qu[j], kc.qs[j]);
        const __m512 f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(d), kc.dk[j], kc.bias[j]);
        acc = _mm512_fmadd_ps(f, qv.dq[j], acc);
    }
    return _mm512_reduce_add_ps(acc);
}

// out must be 64 byte aligned
inline void dequant_q4_0_row(const block_q4_0 * blocks, float * out) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m512i eight = _mm512_set1_epi32(8);
    for (int b = 0; b < NB; ++b) {
        const __m512 d = _mm512_set1_ps(f16_to_f32(blocks[b].d));
        const __m128i raw = _mm_loadu_si128((const __m128i *) blocks[b].qs);
        const __m128i lo = _mm_and_si128(raw, m4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
        const __m512 flo = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(lo), eight));
        const __m512 fhi = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(hi), eight));
        _mm512_store_ps(out + b*QK4_0,           _mm512_mul_ps(flo, d));
        _mm512_store_ps(out + b*QK4_0 + QK4_0/2, _mm512_mul_ps(fhi, d));
    }
}

} // namespace

bool ggml_cuda_kv_stream_cpu_attn_supported() {
#if (defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 9 || (__GNUC__ == 9 && __GNUC_MINOR__ >= 1))) || \
    (defined(__clang__) && __clang_major__ >= 12)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("f16c") &&
        __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

void ggml_cuda_kv_stream_cpu_attn_init(float * out, float * out_meta, int nrows) {
    for (int r = 0; r < nrows; ++r) {
        out_meta[2*r] = -INFINITY;
        out_meta[2*r + 1] = 0.0f;
    }
    std::memset(out, 0, size_t(nrows)*D*sizeof(float));
}

// acc += part; same fold as kv_stream_accumulate_chunk_results with nparts = 1
// a blind side is skipped, not weighted by zero: 0*NaN would corrupt a live row
void ggml_cuda_kv_stream_cpu_attn_fold(
        float * acc, float * acc_meta, const float * part, const float * part_meta, int nrows) {
    for (int r = 0; r < nrows; ++r) {
        const float m_part = part_meta[2*r];
        if (m_part == -INFINITY) {
            continue;
        }
        float * a = acc + size_t(r)*D;
        const float * b = part + size_t(r)*D;
        const float m_old = acc_meta[2*r];
        if (m_old == -INFINITY) {
            acc_meta[2*r] = m_part;
            acc_meta[2*r + 1] = part_meta[2*r + 1];
            std::memcpy(a, b, D*sizeof(float));
            continue;
        }
        const float m = std::max(m_old, m_part);
        const float w_old = std::exp(m_old - m);
        const float w_part = std::exp(m_part - m);
        acc_meta[2*r] = m;
        acc_meta[2*r + 1] = w_old*acc_meta[2*r + 1] + w_part*part_meta[2*r + 1];
        for (int i = 0; i < D; ++i) {
            a[i] = w_old*a[i] + w_part*b[i];
        }
    }
}

void ggml_cuda_kv_stream_cpu_attn(const ggml_cuda_kv_stream_cpu_attn_params & p) {
    GGML_ASSERT(p.n_tokens > 0 && p.n_tokens <= MAX_TOKENS);
    GGML_ASSERT(p.n_head > 0 && p.n_head_kv > 0 && p.n_head%p.n_head_kv == 0);
    GGML_ASSERT(p.page_tokens > 0 && p.page_tokens%TILE == 0);
    GGML_ASSERT(p.k && p.v && p.q && p.out && p.out_meta && (p.n_pages == 0 || p.pages));

    const int group = p.n_head/p.n_head_kv;

    std::vector<q_vec> qv(size_t(p.n_tokens)*p.n_head);
    for (int t = 0; t < p.n_tokens; ++t) {
        for (int h = 0; h < p.n_head; ++h) {
            quantize_q((const float *)((const uint8_t *) p.q + size_t(t)*p.q_token_stride + size_t(h)*p.q_head_stride),
                qv[size_t(t*p.n_head + h)]);
        }
    }

    alignas(64) k_cell kc[TILE];
    alignas(64) float vt[TILE][D];
    const __m512 neg_inf = _mm512_set1_ps(-INFINITY);
    const __m512 vscale = _mm512_set1_ps(p.scale);
    __m512 mrow[MAX_TOKENS];
    bool alive[MAX_TOKENS];

    // mask is the same for every KV head, so decode once and reuse the tile
    for (uint32_t slot = 0; slot < p.n_pages; ++slot) {
        for (uint32_t c0 = 0; c0 < p.page_tokens; c0 += TILE) {
            const size_t cell0 = size_t(p.pages[slot])*p.page_tokens + c0;
            const size_t col0 = size_t(slot)*p.page_tokens + c0;

            bool any = false;
            for (int t = 0; t < p.n_tokens; ++t) {
                __m512 m = _mm512_setzero_ps();
                if (p.mask != nullptr) {
                    const uint16_t * mp = (const uint16_t *)((const uint8_t *) p.mask + size_t(t)*p.mask_token_stride) + col0;
                    m = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) mp));
                }
                mrow[t] = m;
                alive[t] = _mm512_cmp_ps_mask(m, neg_inf, _CMP_GT_OQ) != 0;
                any = any || alive[t];
            }
            if (!any) { continue; }

            for (int kvh = 0; kvh < p.n_head_kv; ++kvh) {
                for (int c = 0; c < TILE; ++c) {
                    prepare_k((const block_q8_0 *)(p.k + (cell0 + c)*p.k_token_stride + size_t(kvh)*p.k_head_stride), kc[c]);
                    dequant_q4_0_row((const block_q4_0 *)(p.v + (cell0 + c)*p.v_token_stride + size_t(kvh)*p.v_head_stride), vt[c]);
                }

                for (int t = 0; t < p.n_tokens; ++t) {
                    if (!alive[t]) { continue; }
                    for (int g = 0; g < group; ++g) {
                        const int r = t*p.n_head + kvh*group + g;
                        alignas(64) float s[TILE];
                        for (int c = 0; c < TILE; ++c) { s[c] = score(qv[size_t(r)], kc[c]); }

                        float & M = p.out_meta[2*r];
                        float & S = p.out_meta[2*r + 1];
                        const __m512 sv = _mm512_fmadd_ps(_mm512_load_ps(s), vscale, mrow[t]);
                        const float m_new = std::max(M, _mm512_reduce_max_ps(sv));
                        const __m512 pv = v_expf(_mm512_sub_ps(sv, _mm512_set1_ps(m_new)));
                        const float w_old = M == -INFINITY ? 0.0f : std::exp(M - m_new);
                        S = w_old*S + _mm512_reduce_add_ps(pv);
                        M = m_new;

                        alignas(64) float pr[TILE];
                        _mm512_store_ps(pr, pv);
                        float * vkq = p.out + size_t(r)*D;
                        const __m512 w = _mm512_set1_ps(w_old);
                        __m512 acc[D/16];
                        for (int i = 0; i < D/16; ++i) { acc[i] = _mm512_mul_ps(_mm512_loadu_ps(vkq + 16*i), w); }
                        for (int c = 0; c < TILE; ++c) {
                            const __m512 pc = _mm512_set1_ps(pr[c]);
                            for (int i = 0; i < D/16; ++i) {
                                acc[i] = _mm512_fmadd_ps(pc, _mm512_load_ps(vt[c] + 16*i), acc[i]);
                            }
                        }
                        for (int i = 0; i < D/16; ++i) { _mm512_storeu_ps(vkq + 16*i, acc[i]); }
                    }
                }
            }
        }
    }
}

#else

bool ggml_cuda_kv_stream_cpu_attn_supported() {
    return false;
}

void ggml_cuda_kv_stream_cpu_attn_init(float * out, float * out_meta, int nrows) {
    GGML_UNUSED_VARS(out, out_meta, nrows);
    GGML_ABORT("CPU split attention needs an x86-64 AVX-512 build");
}

void ggml_cuda_kv_stream_cpu_attn_fold(
        float * acc, float * acc_meta, const float * part, const float * part_meta, int nrows) {
    GGML_UNUSED_VARS(acc, acc_meta, part, part_meta, nrows);
    GGML_ABORT("CPU split attention needs an x86-64 AVX-512 build");
}

void ggml_cuda_kv_stream_cpu_attn(const ggml_cuda_kv_stream_cpu_attn_params & params) {
    GGML_UNUSED_VARS(params);
    GGML_ABORT("CPU split attention needs an x86-64 AVX-512 build");
}

#endif // GGML_CUDA_KV_STREAM_CPU_ATTN_AVX512
