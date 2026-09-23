#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-cuda/kv-stream-span-tuner.h"
#include "../ggml/src/ggml-cuda/kv-stream-cpu-attn.h"
#include "../ggml/src/ggml-cuda/kv-stream-cpu-pages.h"
#include "../ggml/src/ggml-cuda/kv-stream-geometry.h"
#include "ggml.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cfloat>
#include <random>
#include <vector>

namespace {

constexpr int64_t HEAD_DIM = GGML_CUDA_KV_STREAM_HEAD_DIM;
constexpr int64_t N_KV_HEAD = 4;
constexpr int64_t N_Q_HEAD  = 24;

float max_abs_error(const std::vector<float> & expected, const std::vector<float> & actual) {
    GGML_ASSERT(expected.size() == actual.size());
    float max_abs = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
    }
    return max_abs;
}

bool is_finite(const std::vector<float> & values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

struct attention_inputs {
    ggml_type type_k = GGML_TYPE_Q8_0;
    ggml_type type_v = GGML_TYPE_Q4_0;
    std::vector<float> q;
    std::vector<uint8_t> k;
    std::vector<uint8_t> v;
    std::vector<uint16_t> mask;
};

attention_inputs make_inputs(
        int64_t n_kv, int64_t n_batch, int64_t query_start = 0,
        ggml_type type_k = GGML_TYPE_Q8_0, ggml_type type_v = GGML_TYPE_Q4_0,
        float phase = 0.0f) {
    attention_inputs result;
    result.type_k = type_k;
    result.type_v = type_v;

    result.q.resize(HEAD_DIM*n_batch*N_Q_HEAD);
    for (size_t i = 0; i < result.q.size(); ++i) {
        result.q[i] = 0.15f*std::sin(float(i)*0.03125f) + 0.05f*std::cos(float(i)*0.0078125f);
    }

    const int64_t nrows = n_kv*N_KV_HEAD;
    std::vector<float> source(HEAD_DIM*nrows);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = 0.4f*std::sin(float(i)*0.001953125f + phase) + 0.2f*std::cos(float(i)*0.00048828125f + phase);
    }

    result.k.resize(ggml_row_size(type_k, HEAD_DIM)*nrows);
    const size_t k_written = ggml_quantize_chunk(
        type_k, source.data(), result.k.data(), 0, nrows, HEAD_DIM, nullptr);
    GGML_ASSERT(k_written == result.k.size());

    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = 0.35f*std::cos(float(i)*0.00146484375f + phase) - 0.1f*std::sin(float(i)*0.00390625f + phase);
    }
    result.v.resize(ggml_row_size(type_v, HEAD_DIM)*nrows);
    const size_t v_written = ggml_quantize_chunk(
        type_v, source.data(), result.v.data(), 0, nrows, HEAD_DIM, nullptr);
    GGML_ASSERT(v_written == result.v.size());

    result.mask.resize(n_kv*n_batch);
    for (int64_t batch = 0; batch < n_batch; ++batch) {
        for (int64_t token = 0; token < n_kv; ++token) {
            // vary both token blocks and query rows to catch a wrong mask row pitch
            const float bias = token <= query_start + batch ?
                -0.015625f*float((token + 73*batch) % 127) : -INFINITY;
            result.mask[batch*n_kv + token] = ggml_fp32_to_fp16(bias);
        }
    }
    return result;
}

attention_inputs make_f16_reference(const attention_inputs & inputs, int64_t n_kv) {
    attention_inputs result;
    result.type_k = GGML_TYPE_F16;
    result.type_v = GGML_TYPE_F16;
    result.q = inputs.q;
    result.mask = inputs.mask;

    const int64_t nrows = n_kv*N_KV_HEAD;
    result.k.resize(ggml_row_size(GGML_TYPE_F16, HEAD_DIM)*nrows);
    result.v.resize(ggml_row_size(GGML_TYPE_F16, HEAD_DIM)*nrows);
    std::vector<float> row(HEAD_DIM);

    auto convert = [&](ggml_type type, const std::vector<uint8_t> & source,
            std::vector<uint8_t> & destination) {
        const size_t source_row_bytes = ggml_row_size(type, HEAD_DIM);
        const size_t destination_row_bytes = ggml_row_size(GGML_TYPE_F16, HEAD_DIM);
        const auto * traits = ggml_get_type_traits(type);
        for (int64_t i = 0; i < nrows; ++i) {
            const void * source_row = source.data() + size_t(i)*source_row_bytes;
            if (type == GGML_TYPE_F32) {
                std::memcpy(row.data(), source_row, HEAD_DIM*sizeof(float));
            } else {
                GGML_ASSERT(traits->to_float != nullptr);
                traits->to_float(source_row, row.data(), HEAD_DIM);
            }
            ggml_fp32_to_fp16_row(
                row.data(),
                reinterpret_cast<ggml_fp16_t *>(
                    destination.data() + size_t(i)*destination_row_bytes),
                HEAD_DIM);
        }
    };

    convert(inputs.type_k, inputs.k, result.k);
    convert(inputs.type_v, inputs.v, result.v);
    return result;
}

constexpr uint32_t PAGE_TOKENS = GGML_CUDA_KV_STREAM_PAGE_TOKENS;

size_t query_page_bytes(ggml_backend_t backend, ggml_type type_k, ggml_type type_v) {
    using page_bytes_fn_t = bool (*)(
        ggml_type, ggml_type, uint32_t, uint32_t, uint32_t, uint32_t, size_t *);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto page_bytes_fn = reinterpret_cast<page_bytes_fn_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_page_bytes"));
    size_t page_bytes = 0;
    GGML_ASSERT(page_bytes_fn != nullptr && page_bytes_fn(
        type_k, type_v, HEAD_DIM, HEAD_DIM, N_KV_HEAD, PAGE_TOKENS, &page_bytes));
    return page_bytes;
}

size_t query_conversion_bytes(ggml_backend_t backend, ggml_type type_k, ggml_type type_v) {
    using workspace_fn_t = bool (*)(
        ggml_type, ggml_type, uint32_t, uint32_t, uint32_t, uint32_t, size_t *);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto workspace_fn = reinterpret_cast<workspace_fn_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_workspace_bytes"));
    size_t workspace_bytes = 0;
    GGML_ASSERT(workspace_fn != nullptr && workspace_fn(
        type_k, type_v, HEAD_DIM, HEAD_DIM, N_KV_HEAD, PAGE_TOKENS, &workspace_bytes));
    return workspace_bytes;
}

struct alignas(64) cpu_attn_line { uint8_t bytes[64]; };

std::vector<cpu_attn_line> cpu_attn_wdata(int n_tokens, int n_head) {
    return std::vector<cpu_attn_line>((size_t(n_tokens)*n_head*GGML_CUDA_KV_STREAM_CPU_ATTN_ROW_WSIZE + 63)/64);
}

struct cpu_attn_case {
    attention_inputs inputs;
    std::vector<uint32_t> pages;
    std::vector<uint16_t> mask;
    std::vector<float> out, out_ref, meta, meta_ref;
    std::vector<cpu_attn_line> wdata;
    int64_t n_batch = 0;
    int64_t n_head = N_Q_HEAD;
    int64_t n_head_kv = N_KV_HEAD;
    size_t mask_width = 0;

    cpu_attn_case(int64_t n_kv, int64_t n_batch_, std::vector<uint32_t> pages_)
            : inputs(make_inputs(n_kv, n_batch_, n_kv)), pages(std::move(pages_)), n_batch(n_batch_) {
        GGML_ASSERT(inputs.type_k == GGML_TYPE_Q8_0 && inputs.type_v == GGML_TYPE_Q4_0);
        mask_width = pages.size()*PAGE_TOKENS;
        set_heads(N_Q_HEAD, N_KV_HEAD);
    }

    int rows() const { return int(n_batch*n_head); }

    // the first heads of Q are reused, so the GQA group can be narrowed down to MHA
    void set_heads(int64_t heads, int64_t heads_kv) {
        n_head = heads;
        n_head_kv = heads_kv;
        out.assign(size_t(rows())*HEAD_DIM, 0.0f);
        out_ref = out;
        meta.assign(size_t(rows())*2, 0.0f);
        meta_ref = meta;
        wdata = cpu_attn_wdata(int(n_batch), int(n_head));
    }

    // visible(token, slot, cell) -> bias or -inf
    template <typename F>
    void set_mask(F visible) {
        mask_width = pages.size()*PAGE_TOKENS;
        mask.resize(size_t(n_batch)*mask_width);
        for (int64_t t = 0; t < n_batch; ++t) {
            for (size_t slot = 0; slot < pages.size(); ++slot) {
                for (uint32_t c = 0; c < PAGE_TOKENS; ++c) {
                    mask[size_t(t)*mask_width + slot*PAGE_TOKENS + c] = ggml_fp32_to_fp16(visible(t, slot, c));
                }
            }
        }
    }

    // the mask spans the whole cache, not only the pages this call attends
    void set_cache_wide_mask(int64_t n_kv) {
        mask = inputs.mask;
        mask_width = size_t(n_kv);
    }

    ggml_cuda_kv_stream_cpu_attn_params make(float * o, float * m, size_t slot_begin, size_t slot_end) {
        ggml_cuda_kv_stream_cpu_attn_params p;
        p.k = inputs.k.data();
        p.k_token_stride = ggml_row_size(inputs.type_k, HEAD_DIM)*N_KV_HEAD;
        p.k_head_stride = ggml_row_size(inputs.type_k, HEAD_DIM);
        p.v = inputs.v.data();
        p.v_token_stride = ggml_row_size(inputs.type_v, HEAD_DIM)*N_KV_HEAD;
        p.v_head_stride = ggml_row_size(inputs.type_v, HEAD_DIM);
        p.q = inputs.q.data();
        p.q_token_stride = HEAD_DIM*sizeof(float);
        p.q_head_stride = HEAD_DIM*n_batch*sizeof(float);
        p.mask = mask.empty() ? nullptr : mask.data() + slot_begin*PAGE_TOKENS;
        p.mask_token_stride = mask_width*sizeof(uint16_t);
        p.pages = pages.data() + slot_begin;
        p.n_pages = uint32_t(slot_end - slot_begin);
        p.page_tokens = PAGE_TOKENS;
        p.n_head = int(n_head);
        p.n_head_kv = int(n_head_kv);
        p.n_tokens = int(n_batch);
        p.scale = 1.0f/16.0f;
        p.out = o;
        p.out_meta = m;
        p.wdata = wdata.data();
        p.wsize = wdata.size()*sizeof(cpu_attn_line);
        return p;
    }
};

void cpu_attn_quantize_q(const float * x, std::vector<float> & q) {
    for (int b = 0; b < HEAD_DIM/32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(x[b*32 + i])); }
        const float d = amax/127.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;
        for (int i = 0; i < 32; ++i) { q[b*32 + i] = d*float(std::lrintf(x[b*32 + i]*id)); }
    }
}

void cpu_attn_reference(const ggml_cuda_kv_stream_cpu_attn_params & p) {
    const auto * k_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    const auto * v_traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
    const int group = p.n_head/p.n_head_kv;
    const int64_t n_cells = int64_t(p.n_pages)*p.page_tokens;
    std::vector<float> qf(HEAD_DIM), kf(HEAD_DIM), vf(HEAD_DIM), s(n_cells);
    for (int t = 0; t < p.n_tokens; ++t) {
        for (int h = 0; h < p.n_head; ++h) {
            const int r = t*p.n_head + h;
            const int kvh = h/group;
            cpu_attn_quantize_q((const float *)((const uint8_t *) p.q + size_t(t)*p.q_token_stride + size_t(h)*p.q_head_stride), qf);
            float M = -INFINITY;
            for (int64_t i = 0; i < n_cells; ++i) {
                const size_t cell = size_t(p.pages[i/p.page_tokens])*p.page_tokens + size_t(i%p.page_tokens);
                const float m = p.mask == nullptr ? 0.0f :
                    ggml_fp16_to_fp32(*((const uint16_t *)((const uint8_t *) p.mask + size_t(t)*p.mask_token_stride) + i));
                if (m == -INFINITY) { s[i] = -INFINITY; continue; }
                k_traits->to_float(p.k + cell*p.k_token_stride + size_t(kvh)*p.k_head_stride, kf.data(), HEAD_DIM);
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) { dot += qf[d]*kf[d]; }
                s[i] = p.scale*dot + m;
                M = std::max(M, s[i]);
            }
            if (M == -INFINITY) { continue; }
            float S = 0.0f;
            float * vkq = p.out + size_t(r)*HEAD_DIM;
            for (int64_t i = 0; i < n_cells; ++i) {
                if (s[i] == -INFINITY) { continue; }
                const size_t cell = size_t(p.pages[i/p.page_tokens])*p.page_tokens + size_t(i%p.page_tokens);
                const float w = std::exp(s[i] - M);
                S += w;
                v_traits->to_float(p.v + cell*p.v_token_stride + size_t(kvh)*p.v_head_stride, vf.data(), HEAD_DIM);
                for (int d = 0; d < HEAD_DIM; ++d) { vkq[d] += w*vf[d]; }
            }
            p.out_meta[2*r] = M;
            p.out_meta[2*r + 1] = S;
        }
    }
}

bool cpu_attn_same_rows(testing & t, const char * name, int rows,
        const std::vector<float> & out, const std::vector<float> & meta,
        const std::vector<float> & out_ref, const std::vector<float> & meta_ref) {
    double max_err = 0.0, max_lse_err = 0.0;
    int bad_rows = 0;
    for (int r = 0; r < rows; ++r) {
        const float M = meta[2*r], S = meta[2*r + 1];
        const float Mr = meta_ref[2*r], Sr = meta_ref[2*r + 1];
        // a row blind on one side only is a mismatch, and dividing by its S would be nan
        if (M == -INFINITY || Mr == -INFINITY) {
            if (M != Mr || S != 0.0f) { ++bad_rows; }
            continue;
        }
        const double lse = double(M) + std::log(double(S));
        const double lse_ref = double(Mr) + std::log(double(Sr));
        max_lse_err = std::max(max_lse_err, std::fabs(lse - lse_ref)/std::max(1.0, std::fabs(lse_ref)));
        for (int d = 0; d < HEAD_DIM; ++d) {
            max_err = std::max(max_err, std::fabs(double(out[size_t(r)*HEAD_DIM + d])/S - double(out_ref[size_t(r)*HEAD_DIM + d])/Sr));
        }
    }
    const bool same = bad_rows == 0 && max_err < 1e-4 && max_lse_err < 1e-4;
    if (!same) {
        std::fprintf(stderr, "%s: blind rows=%d max |out/S| err=%g max lse err=%g\n",
            name, bad_rows, max_err, max_lse_err);
    }
    return t.assert_true(name, same);
}

bool cpu_attn_same(testing & t, const char * name, const cpu_attn_case & c) {
    return cpu_attn_same_rows(t, name, c.rows(), c.out, c.meta, c.out_ref, c.meta_ref);
}

void cpu_attn_check(testing & t, cpu_attn_case & c, const char * name, size_t slot_begin, size_t slot_end) {
    const int rows = c.rows();
    ggml_cuda_kv_stream_cpu_attn_init(c.out_ref.data(), c.meta_ref.data(), rows);
    cpu_attn_reference(c.make(c.out_ref.data(), c.meta_ref.data(), slot_begin, slot_end));
    ggml_cuda_kv_stream_cpu_attn_init(c.out.data(), c.meta.data(), rows);
    ggml_cuda_kv_stream_cpu_attn(c.make(c.out.data(), c.meta.data(), slot_begin, slot_end));
    cpu_attn_same(t, name, c);
}

std::vector<float> run_attention(
        ggml_backend_t backend,
        const attention_inputs & inputs,
        ggml_backend_buffer_type_t kv_buft,
        int64_t n_kv,
        int64_t n_batch,
        int repeats = 1,
        int64_t update_rows = 1,
        bool change_updates = false,
        ggml_type index_type = GGML_TYPE_I32,
        bool index_on_host = true,
        ggml_backend_cuda_kv_stream_runtime_t dirty_runtime = nullptr,
        bool change_indices = false,
        bool replace_cache = false,
        uint64_t graph_uid = 0,
        const int64_t * custom_rows = nullptr) {
    constexpr size_t N_TENSORS = 32;
    const size_t context_bytes = ggml_tensor_overhead()*N_TENSORS + ggml_graph_overhead_custom(N_TENSORS, false);

    ggml_init_params params{
        /* .mem_size   = */ context_bytes,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr compute_ctx(ggml_init(params));
    ggml_context_ptr kv_ctx(ggml_init(params));
    ggml_context_ptr index_ctx(ggml_init(params));
    GGML_ASSERT(compute_ctx && kv_ctx && index_ctx);

    ggml_tensor * q = ggml_new_tensor_4d(
        compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM, n_batch, N_Q_HEAD, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(
        compute_ctx.get(), GGML_TYPE_F16, n_kv, n_batch, 1, 1);
    ggml_tensor * k_storage = ggml_new_tensor_2d(
        kv_ctx.get(), inputs.type_k, HEAD_DIM*N_KV_HEAD, n_kv);
    ggml_tensor * v_storage = ggml_new_tensor_2d(
        kv_ctx.get(), inputs.type_v, HEAD_DIM*N_KV_HEAD, n_kv);
    ggml_tensor * k_cache = ggml_view_4d(
        kv_ctx.get(), k_storage, HEAD_DIM, N_KV_HEAD, n_kv, 1,
        ggml_row_size(inputs.type_k, HEAD_DIM),
        ggml_row_size(inputs.type_k, HEAD_DIM*N_KV_HEAD),
        ggml_row_size(inputs.type_k, HEAD_DIM*N_KV_HEAD)*n_kv, 0);
    ggml_tensor * v_cache = ggml_view_4d(
        kv_ctx.get(), v_storage, HEAD_DIM, N_KV_HEAD, n_kv, 1,
        ggml_row_size(inputs.type_v, HEAD_DIM),
        ggml_row_size(inputs.type_v, HEAD_DIM*N_KV_HEAD),
        ggml_row_size(inputs.type_v, HEAD_DIM*N_KV_HEAD)*n_kv, 0);
    ggml_tensor * k = ggml_permute(kv_ctx.get(), k_cache, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(kv_ctx.get(), v_cache, 0, 2, 1, 3);

    ggml_tensor * k_update = ggml_new_tensor_2d(
        compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM*N_KV_HEAD, update_rows);
    ggml_tensor * v_update = ggml_new_tensor_2d(
        compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM*N_KV_HEAD, update_rows);
    ggml_tensor * update_index = ggml_new_tensor_1d(index_ctx.get(), index_type, update_rows);
    ggml_tensor * updated_k = ggml_set_rows(compute_ctx.get(), k_storage, k_update, update_index);
    ggml_tensor * updated_v = ggml_set_rows(compute_ctx.get(), v_storage, v_update, update_index);

    ggml_tensor * out = ggml_flash_attn_ext(
        compute_ctx.get(), q, k, v, mask, 1.0f/std::sqrt(float(HEAD_DIM)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_set_name(out, "streamed-attention-output");

    ggml_backend_buffer_ptr kv_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(kv_ctx.get(), kv_buft));
    ggml_backend_buffer_ptr compute_buffer(
        ggml_backend_alloc_ctx_tensors(compute_ctx.get(), backend));
    ggml_backend_buffer_ptr index_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(index_ctx.get(), index_on_host ?
            ggml_backend_cuda_host_buffer_type() : ggml_backend_get_default_buffer_type(backend)));
    GGML_ASSERT(kv_buffer && compute_buffer && index_buffer);

    ggml_backend_tensor_set(q, inputs.q.data(), 0, inputs.q.size()*sizeof(float));
    ggml_backend_tensor_set(k_storage, inputs.k.data(), 0, inputs.k.size());
    ggml_backend_tensor_set(v_storage, inputs.v.data(), 0, inputs.v.size());
    ggml_backend_tensor_set(mask, inputs.mask.data(), 0, inputs.mask.size()*sizeof(uint16_t));

    std::vector<float> k_update_data(HEAD_DIM*N_KV_HEAD*update_rows);
    std::vector<float> v_update_data(HEAD_DIM*N_KV_HEAD*update_rows);
    for (size_t i = 0; i < k_update_data.size(); ++i) {
        k_update_data[i] = 0.6f*std::sin(float(i)*0.0234375f);
        v_update_data[i] = 0.4f*std::cos(float(i)*0.017578125f);
    }
    ggml_backend_tensor_set(k_update, k_update_data.data(), 0, k_update_data.size()*sizeof(float));
    ggml_backend_tensor_set(v_update, v_update_data.data(), 0, v_update_data.size()*sizeof(float));
    std::vector<int64_t> dirty_rows(update_rows);
    for (int64_t row = 0; row < update_rows; ++row) {
        dirty_rows[row] = custom_rows != nullptr ? custom_rows[row] : row;
    }
    if (index_type == GGML_TYPE_I32) {
        std::vector<int32_t> update_index_data(update_rows);
        for (int64_t row = 0; row < update_rows; ++row) { update_index_data[row] = int32_t(dirty_rows[row]); }
        ggml_backend_tensor_set(
            update_index, update_index_data.data(), 0, update_index_data.size()*sizeof(int32_t));
    } else {
        GGML_ASSERT(index_type == GGML_TYPE_I64);
        ggml_backend_tensor_set(
            update_index, dirty_rows.data(), 0, dirty_rows.size()*sizeof(int64_t));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), N_TENSORS, false);
    graph->uid = graph_uid;
    ggml_build_forward_expand(graph, updated_k);
    ggml_build_forward_expand(graph, updated_v);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_supports_op(backend, updated_k));
    GGML_ASSERT(ggml_backend_supports_op(backend, updated_v));
    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "unsupported attention K=%s V=%s buft=%s\n",
            ggml_type_name(inputs.type_k), ggml_type_name(inputs.type_v),
            ggml_backend_buft_name(kv_buft));
    }
    GGML_ASSERT(ggml_backend_supports_op(backend, out));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (replace_cache && repeat == 3) {
            std::vector<uint8_t> zero_k(inputs.k.size(), 0);
            std::vector<uint8_t> zero_v(inputs.v.size(), 0);
            ggml_backend_tensor_set(k_storage, zero_k.data(), 0, zero_k.size());
            ggml_backend_tensor_set(v_storage, zero_v.data(), 0, zero_v.size());
        }
        if (change_indices) {
            GGML_ASSERT(index_type == GGML_TYPE_I64);
            for (int64_t row = 0; row < update_rows; ++row) {
                dirty_rows[row] = (int64_t(repeat)*update_rows + row)%n_kv;
            }
            ggml_backend_tensor_set(
                update_index, dirty_rows.data(), 0, dirty_rows.size()*sizeof(int64_t));
        }
        if (dirty_runtime != nullptr) {
            GGML_ASSERT(ggml_backend_cuda_kv_stream_mark_dirty_rows(
                dirty_runtime, dirty_rows.data(), dirty_rows.size()));
        }
        if (repeat > 0 && change_updates) {
            for (float & value : k_update_data) { value = -2.0f*value; }
            for (float & value : v_update_data) { value = -2.0f*value; }
            ggml_backend_tensor_set(k_update, k_update_data.data(), 0, k_update_data.size()*sizeof(float));
            ggml_backend_tensor_set(v_update, v_update_data.data(), 0, v_update_data.size()*sizeof(float));
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
    return result;
}

std::vector<float> run_attention_layers(
        ggml_backend_t backend,
        const std::vector<attention_inputs> & layers,
        ggml_backend_buffer_type_t kv_buft,
        int64_t n_kv,
        int64_t n_batch,
        int repeats = 1,
        int64_t update_rows = 1,
        ggml_type index_type = GGML_TYPE_I32,
        ggml_backend_cuda_kv_stream_runtime_t dirty_runtime = nullptr,
        uint32_t layout_after_first = 0,
        const int64_t * custom_rows = nullptr,
        bool graph_per_layer = false) {
    constexpr size_t N_TENSORS = 256;
    const size_t graph_count = graph_per_layer ? layers.size() : 1;
    const size_t context_bytes = ggml_tensor_overhead()*N_TENSORS +
        ggml_graph_overhead_custom(N_TENSORS, false)*graph_count;

    ggml_init_params params{
        /* .mem_size   = */ context_bytes,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr compute_ctx(ggml_init(params));
    ggml_context_ptr kv_ctx(ggml_init(params));
    GGML_ASSERT(compute_ctx && kv_ctx);

    struct layer_tensors {
        ggml_tensor * q;
        ggml_tensor * mask;
        ggml_tensor * k_storage;
        ggml_tensor * v_storage;
        ggml_tensor * k_update;
        ggml_tensor * v_update;
        ggml_tensor * update_index;
        ggml_tensor * updated_k;
        ggml_tensor * updated_v;
        ggml_tensor * out;
    };
    std::vector<layer_tensors> tensors;
    tensors.reserve(layers.size());

    for (size_t layer = 0; layer < layers.size(); ++layer) {
        layer_tensors current{};
        current.q = ggml_new_tensor_4d(
            compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM, n_batch, N_Q_HEAD, 1);
        current.mask = ggml_new_tensor_4d(
            compute_ctx.get(), GGML_TYPE_F16, n_kv, n_batch, 1, 1);
        current.k_storage = ggml_new_tensor_2d(
            kv_ctx.get(), GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD, n_kv);
        current.v_storage = ggml_new_tensor_2d(
            kv_ctx.get(), GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD, n_kv);
        ggml_tensor * k_cache = ggml_view_4d(
            kv_ctx.get(), current.k_storage, HEAD_DIM, N_KV_HEAD, n_kv, 1,
            ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM),
            ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD),
            ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD)*n_kv, 0);
        ggml_tensor * v_cache = ggml_view_4d(
            kv_ctx.get(), current.v_storage, HEAD_DIM, N_KV_HEAD, n_kv, 1,
            ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM),
            ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD),
            ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD)*n_kv, 0);
        ggml_tensor * k = ggml_permute(kv_ctx.get(), k_cache, 0, 2, 1, 3);
        ggml_tensor * v = ggml_permute(kv_ctx.get(), v_cache, 0, 2, 1, 3);

        current.k_update = ggml_new_tensor_2d(
            compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM*N_KV_HEAD, update_rows);
        current.v_update = ggml_new_tensor_2d(
            compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM*N_KV_HEAD, update_rows);
        current.update_index = ggml_new_tensor_1d(compute_ctx.get(), index_type, update_rows);
        current.updated_k = ggml_set_rows(
            compute_ctx.get(), current.k_storage, current.k_update, current.update_index);
        current.updated_v = ggml_set_rows(
            compute_ctx.get(), current.v_storage, current.v_update, current.update_index);
        current.out = ggml_flash_attn_ext(
            compute_ctx.get(), current.q, k, v, current.mask,
            1.0f/std::sqrt(float(HEAD_DIM)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(current.out, GGML_PREC_F32);
        tensors.push_back(current);
    }

    ggml_backend_buffer_ptr kv_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(kv_ctx.get(), kv_buft));
    ggml_backend_buffer_ptr compute_buffer(
        ggml_backend_alloc_ctx_tensors(compute_ctx.get(), backend));
    GGML_ASSERT(kv_buffer && compute_buffer);

    std::vector<int64_t> dirty_rows(update_rows);
    for (int64_t row = 0; row < update_rows; ++row) {
        dirty_rows[row] = custom_rows != nullptr ? custom_rows[row] : row;
    }
    for (size_t layer = 0; layer < layers.size(); ++layer) {
        const auto & input = layers[layer];
        auto & current = tensors[layer];
        ggml_backend_tensor_set(current.q, input.q.data(), 0, input.q.size()*sizeof(float));
        ggml_backend_tensor_set(current.k_storage, input.k.data(), 0, input.k.size());
        ggml_backend_tensor_set(current.v_storage, input.v.data(), 0, input.v.size());
        ggml_backend_tensor_set(current.mask, input.mask.data(), 0, input.mask.size()*sizeof(uint16_t));

        std::vector<float> k_update_data(HEAD_DIM*N_KV_HEAD*update_rows);
        std::vector<float> v_update_data(HEAD_DIM*N_KV_HEAD*update_rows);
        for (size_t i = 0; i < k_update_data.size(); ++i) {
            k_update_data[i] = (0.5f + 0.03f*layer)*std::sin(float(i)*0.0234375f);
            v_update_data[i] = (0.3f + 0.02f*layer)*std::cos(float(i)*0.017578125f);
        }
        ggml_backend_tensor_set(current.k_update, k_update_data.data(), 0,
            k_update_data.size()*sizeof(float));
        ggml_backend_tensor_set(current.v_update, v_update_data.data(), 0,
            v_update_data.size()*sizeof(float));
        if (index_type == GGML_TYPE_I32) {
            std::vector<int32_t> rows_i32(update_rows);
            for (int64_t row = 0; row < update_rows; ++row) { rows_i32[row] = int32_t(dirty_rows[row]); }
            ggml_backend_tensor_set(
                current.update_index, rows_i32.data(), 0, rows_i32.size()*sizeof(int32_t));
        } else {
            GGML_ASSERT(index_type == GGML_TYPE_I64);
            ggml_backend_tensor_set(
                current.update_index, dirty_rows.data(), 0, dirty_rows.size()*sizeof(int64_t));
        }
    }

    // one graph per ubatch, or one graph per layer when the scheduler splits a ubatch
    std::vector<ggml_cgraph *> graphs;
    for (size_t layer = 0; layer < tensors.size(); ++layer) {
        if (layer == 0 || graph_per_layer) {
            graphs.push_back(ggml_new_graph_custom(compute_ctx.get(), N_TENSORS, false));
        }
        ggml_build_forward_expand(graphs.back(), tensors[layer].updated_k);
        ggml_build_forward_expand(graphs.back(), tensors[layer].updated_v);
        ggml_build_forward_expand(graphs.back(), tensors[layer].out);
    }
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (dirty_runtime != nullptr) {
            GGML_ASSERT(ggml_backend_cuda_kv_stream_mark_dirty_rows(
                dirty_runtime, dirty_rows.data(), dirty_rows.size()));
        }
        for (ggml_cgraph * graph : graphs) {
            GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        }
        if (repeat == 0 && layout_after_first != 0) {
            GGML_ASSERT(ggml_backend_cuda_kv_stream_set_decode_layout(
                dirty_runtime, layout_after_first));
        }
    }

    std::vector<float> result;
    for (auto & current : tensors) {
        const size_t begin = result.size();
        result.resize(begin + ggml_nelements(current.out));
        ggml_backend_tensor_get(current.out, result.data() + begin, 0,
            ggml_nbytes(current.out));
    }
    return result;
}

} // namespace

int main() {
    testing t;

    t.test("decode copy batch tuner selects the faster measured mode per layout", [](testing & t) {
        ggml_cuda_kv_stream_span_tuner production_tuner;
        production_tuner.observe(100.0, /* streamed = */ true, /* greedy = */ false);
        for (uint32_t sample = 0; sample < 4; ++sample) {
            production_tuner.observe(10.0, /* streamed = */ true, /* greedy = */ false);
        }
        t.assert_true("production tuner does not decide from four samples",
            !production_tuner.use_greedy_batch() && !production_tuner.selected());

        ggml_cuda_kv_stream_span_tuner tuner(/* trial_samples = */ 2, 0.005, /* warmup_samples = */ 1);

        tuner.observe(100.0, /* streamed = */ false, /* greedy = */ false);
        t.assert_true("non-streamed graphs do not start a trial", !tuner.use_greedy_batch());
        t.assert_true("non-streamed graphs do not select a mode", !tuner.selected());

        tuner.observe(100.0, /* streamed = */ true, /* greedy = */ false);
        t.assert_true("fixed warmup is not measured", !tuner.use_greedy_batch());

        tuner.observe(10.0, /* streamed = */ true, /* greedy = */ false);
        tuner.observe(10.2, /* streamed = */ true, /* greedy = */ false);
        t.assert_true("tuner advances to greedy trials", tuner.use_greedy_batch());
        t.assert_true("both modes are measured before selection", !tuner.selected());

        tuner.observe(100.0, /* streamed = */ true, /* greedy = */ true);
        t.assert_true("greedy warmup is not measured", !tuner.selected());

        tuner.observe(8.0, /* streamed = */ true, /* greedy = */ true);
        tuner.observe(8.2, /* streamed = */ true, /* greedy = */ true);
        t.assert_true("greedy mode is selected when materially faster", tuner.selected());
        t.assert_true("greedy mode remains active after selection", tuner.use_greedy_batch());

        tuner.reset();
        tuner.observe(100.0, /* streamed = */ true, /* greedy = */ false);
        tuner.observe(10.0, /* streamed = */ true, /* greedy = */ false);
        tuner.observe(10.0, /* streamed = */ true, /* greedy = */ false);
        tuner.observe(100.0, /* streamed = */ true, /* greedy = */ true);
        tuner.observe(10.0, /* streamed = */ true, /* greedy = */ true);
        tuner.observe(10.0, /* streamed = */ true, /* greedy = */ true);
        t.assert_true("tuner selects after both trials", tuner.selected());
        t.assert_true("noise does not displace the fixed copy batch", !tuner.use_greedy_batch());

        tuner.observe(1.0, /* streamed = */ true, /* greedy = */ true);
        t.assert_true("selection remains stable until layout reset", !tuner.use_greedy_batch());
    });

    t.test("cpu page selection takes the highest immutable streamed pages", [](testing & t) {
        using pages = std::vector<uint32_t>;
        const std::vector<uint8_t> all_live;
        const pages no_mutable;

        t.assert_true("the last page stays on the GPU",
            ggml_cuda_kv_stream_select_cpu_pages(8, 2, all_live, no_mutable, false, 1) == pages{6});
        t.assert_true("a one-page layer gets no CPU pages",
            ggml_cuda_kv_stream_select_cpu_pages(1, 0, all_live, no_mutable, false, 4).empty());

        std::vector<uint8_t> live(10, 1);
        live[5] = 0;
        t.assert_true("an oversized request clamps to non-resident, live, immutable pages",
            ggml_cuda_kv_stream_select_cpu_pages(10, 3, live, pages{7}, false, 100) == pages{3, 4, 6, 8});
        t.assert_true("a restore step gets no CPU pages",
            ggml_cuda_kv_stream_select_cpu_pages(10, 3, all_live, no_mutable, true, 5).empty());
        t.assert_true("an interior mutable page is passed over",
            ggml_cuda_kv_stream_select_cpu_pages(10, 2, all_live, pages{7}, false, 2) == pages{6, 8});
    });

    t.test("cpu split scope is q8_0 K, q4_0 V decode at head dim 256", [](testing & t) {
        ggml_init_params params{ggml_tensor_overhead()*32, nullptr, true};
        ggml_context_ptr ctx(ggml_init(params));
        auto attention = [&](int64_t head_dim, int64_t n_tokens, int64_t n_seq, ggml_type type_k, ggml_type type_v) {
            ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, n_tokens, N_Q_HEAD, n_seq);
            ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), type_k, head_dim, 512, N_KV_HEAD, n_seq);
            ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), type_v, head_dim, 512, N_KV_HEAD, n_seq);
            return *ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr, 0.0625f, 0.0f, 0.0f);
        };
        constexpr int64_t n_decode = GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS;

        ggml_tensor decode = attention(HEAD_DIM, n_decode, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        t.assert_true("the widest decode batch", ggml_cuda_kv_stream_cpu_split_supported(&decode));
        ggml_tensor prefill = attention(HEAD_DIM, n_decode + 1, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        t.assert_true("not prefill", !ggml_cuda_kv_stream_cpu_split_supported(&prefill));
        ggml_tensor f16_k = attention(HEAD_DIM, 1, 1, GGML_TYPE_F16, GGML_TYPE_Q4_0);
        t.assert_true("not an f16 K", !ggml_cuda_kv_stream_cpu_split_supported(&f16_k));
        ggml_tensor q8_v = attention(HEAD_DIM, 1, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
        t.assert_true("not a q8_0 V", !ggml_cuda_kv_stream_cpu_split_supported(&q8_v));
        ggml_tensor small_head = attention(128, 1, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        t.assert_true("not head dim 128", !ggml_cuda_kv_stream_cpu_split_supported(&small_head));
        ggml_tensor two_seq = attention(HEAD_DIM, 1, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        t.assert_true("not ne[3] == 2", !ggml_cuda_kv_stream_cpu_split_supported(&two_seq));

        ggml_tensor alibi = decode;
        ggml_set_op_params_f32(&alibi, 1, 8.0f);
        t.assert_true("not a max_bias", !ggml_cuda_kv_stream_cpu_split_supported(&alibi));
        ggml_tensor softcap = decode;
        ggml_set_op_params_f32(&softcap, 2, 30.0f);
        t.assert_true("not a logit_softcap", !ggml_cuda_kv_stream_cpu_split_supported(&softcap));
    });

    t.test("cpu attention kernel matches the scalar reference", [](testing & t) {
        if (!ggml_cuda_kv_stream_cpu_attn_supported()) {
            t.skip("CPU attention needs AVX-512 F, DQ, VNNI, F16C and FMA");
            return;
        }
        constexpr int64_t n_kv = 4*PAGE_TOKENS;
        const float neg_inf = -INFINITY;

        {
            cpu_attn_case c(n_kv, 1, {0, 1, 2, 3});
            cpu_attn_check(t, c, "single token, no mask", 0, 4);
        }
        {
            // three sequences see one page each, the fourth token sees nothing
            cpu_attn_case c(n_kv, 4, {0, 1, 2, 3});
            c.set_mask([&](int64_t token, size_t slot, uint32_t) { return token < 3 && slot == size_t(token) ? 0.0f : neg_inf; });
            cpu_attn_check(t, c, "3 sequences + 1 blind token", 0, 4);
        }
        {
            // finite biases are added, not only -inf; the page list is non-contiguous
            cpu_attn_case c(n_kv, 32, {0, 2, 3});
            c.set_mask([&](int64_t token, size_t, uint32_t cell) { return (cell % 32) != uint32_t(token) ? -0.001f*float(cell % 7) : neg_inf; });
            cpu_attn_check(t, c, "32 tokens, biased mask, pages 0 2 3", 0, 3);
        }
        {
            // MHA: every Q head has its own K head, so the group collapses to 1
            cpu_attn_case c(n_kv, 5, {3, 1});
            c.set_heads(N_KV_HEAD, N_KV_HEAD);
            c.set_mask([&](int64_t token, size_t slot, uint32_t cell) { return cell + token < 200 || slot == 1 ? -0.03125f*float(cell % 11) : neg_inf; });
            cpu_attn_check(t, c, "MHA group of one", 0, 2);
        }
        {
            // a Q head of zeros quantizes with d = 0 and leaves every score tied on the mask
            cpu_attn_case c(n_kv, 2, {1, 0});
            std::fill(c.inputs.q.begin() + HEAD_DIM*c.n_batch, c.inputs.q.begin() + 2*HEAD_DIM*c.n_batch, 0.0f);
            c.set_mask([&](int64_t, size_t slot, uint32_t cell) { return slot == 0 && cell >= 128 ? neg_inf : 0.0f; });
            cpu_attn_check(t, c, "zero Q head, tied scores", 0, 2);
        }
        {
            // fold(kernel[slots 0-1], kernel[slots 2-3]) == reference[all]
            cpu_attn_case c(n_kv, 4, {0, 1, 2, 3});
            c.set_mask([&](int64_t token, size_t slot, uint32_t) {
                if (token == 0) { return 0.0f; }
                if (token == 1) { return slot < 2 ? 0.0f : neg_inf; }
                if (token == 2) { return slot >= 2 ? 0.0f : neg_inf; }
                return neg_inf;
            });
            const int rows = c.rows();
            ggml_cuda_kv_stream_cpu_attn_init(c.out_ref.data(), c.meta_ref.data(), rows);
            cpu_attn_reference(c.make(c.out_ref.data(), c.meta_ref.data(), 0, 4));
            std::vector<float> part(c.out.size()), part_meta(c.meta.size());
            ggml_cuda_kv_stream_cpu_attn_init(c.out.data(), c.meta.data(), rows);
            ggml_cuda_kv_stream_cpu_attn(c.make(c.out.data(), c.meta.data(), 0, 2));
            ggml_cuda_kv_stream_cpu_attn_init(part.data(), part_meta.data(), rows);
            ggml_cuda_kv_stream_cpu_attn(c.make(part.data(), part_meta.data(), 2, 4));
            ggml_cuda_kv_stream_cpu_attn_fold(c.out.data(), c.meta.data(), part.data(), part_meta.data(), rows);
            cpu_attn_same(t, "fold of two kernel halves", c);
        }
    });

    t.test("cpu attention kernel holds over random shapes and fold splits", [](testing & t) {
        if (!ggml_cuda_kv_stream_cpu_attn_supported()) {
            t.skip("CPU attention needs AVX-512 F, DQ, VNNI, F16C and FMA");
            return;
        }
        std::mt19937 rng(20240914);
        auto pick = [&](uint32_t n) { return rng()%n; };
        std::vector<cpu_attn_line> wdata = cpu_attn_wdata(GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS, 4*8);

        for (int iter = 0; iter < 64; ++iter) {
            const int n_head_kv = 1 + int(pick(4));
            const int n_head = n_head_kv*(1 + int(pick(8)));
            const int n_tokens = 1 + int(pick(GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS));
            const uint32_t page_tokens = (1 + pick(16))*16;
            const uint32_t n_pages = pick(4);          // 0 must leave the accumulator alone
            const uint32_t cache_pages = n_pages + 1 + pick(2);
            const float scale = std::pow(2.0f, float(int(pick(9)) - 4));
            // padded row pitches: the kernel must not assume heads are packed end to end
            const size_t k_row = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM);
            const size_t v_row = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM);
            const size_t k_token_stride = k_row*n_head_kv + 64*pick(3);
            const size_t v_token_stride = v_row*n_head_kv + 64*pick(3);

            std::vector<uint32_t> pages(n_pages);
            for (auto & page : pages) { page = pick(cache_pages); }   // duplicates are allowed

            const int64_t cells = int64_t(cache_pages)*page_tokens;
            std::vector<float> source(size_t(cells)*n_head_kv*HEAD_DIM);
            for (size_t i = 0; i < source.size(); ++i) { source[i] = std::sin(float(i)*0.01953125f); }
            std::vector<uint8_t> k(size_t(cells)*k_token_stride), v(size_t(cells)*v_token_stride);
            for (int64_t cell = 0; cell < cells; ++cell) {
                const float * row = source.data() + size_t(cell)*n_head_kv*HEAD_DIM;
                ggml_quantize_chunk(GGML_TYPE_Q8_0, row, k.data() + size_t(cell)*k_token_stride, 0, n_head_kv, HEAD_DIM, nullptr);
                ggml_quantize_chunk(GGML_TYPE_Q4_0, row, v.data() + size_t(cell)*v_token_stride, 0, n_head_kv, HEAD_DIM, nullptr);
            }

            std::vector<float> q(size_t(n_tokens)*n_head*HEAD_DIM);
            for (size_t i = 0; i < q.size(); ++i) { q[i] = 0.2f*std::cos(float(i)*0.0078125f); }

            const size_t width = size_t(n_pages)*page_tokens;
            std::vector<uint16_t> mask(size_t(n_tokens)*std::max<size_t>(width, 1));
            for (size_t i = 0; i < mask.size(); ++i) {
                const uint32_t kind = pick(10);
                mask[i] = ggml_fp32_to_fp16(kind < 4 ? -INFINITY : -0.25f*float(pick(16)));
            }

            ggml_cuda_kv_stream_cpu_attn_params p;
            p.k = k.data(); p.k_token_stride = k_token_stride; p.k_head_stride = k_row;
            p.v = v.data(); p.v_token_stride = v_token_stride; p.v_head_stride = v_row;
            p.q = q.data(); p.q_token_stride = HEAD_DIM*sizeof(float);
            p.q_head_stride = size_t(HEAD_DIM)*n_tokens*sizeof(float);
            p.mask = mask.data(); p.mask_token_stride = width*sizeof(uint16_t);
            p.pages = pages.data(); p.n_pages = n_pages; p.page_tokens = page_tokens;
            p.n_head = n_head; p.n_head_kv = n_head_kv; p.n_tokens = n_tokens; p.scale = scale;
            p.wdata = wdata.data(); p.wsize = wdata.size()*sizeof(cpu_attn_line);

            const int rows = n_tokens*n_head;
            std::vector<float> out(size_t(rows)*HEAD_DIM), meta(size_t(rows)*2);
            std::vector<float> out_ref(out.size()), meta_ref(meta.size());

            ggml_cuda_kv_stream_cpu_attn_init(out_ref.data(), meta_ref.data(), rows);
            p.out = out_ref.data(); p.out_meta = meta_ref.data();
            cpu_attn_reference(p);

            // split the page list in two and fold, which also covers the whole span at split 0
            const uint32_t split = n_pages == 0 ? 0 : pick(n_pages + 1);
            std::vector<float> part(out.size()), part_meta(meta.size());
            ggml_cuda_kv_stream_cpu_attn_init(out.data(), meta.data(), rows);
            ggml_cuda_kv_stream_cpu_attn_init(part.data(), part_meta.data(), rows);
            if (split > 0) {
                p.out = out.data(); p.out_meta = meta.data();
                p.pages = pages.data(); p.n_pages = split; p.mask = mask.data();
                ggml_cuda_kv_stream_cpu_attn(p);
            }
            if (split < n_pages) {
                p.out = part.data(); p.out_meta = part_meta.data();
                p.pages = pages.data() + split; p.n_pages = n_pages - split;
                p.mask = mask.data() + size_t(split)*page_tokens;
                ggml_cuda_kv_stream_cpu_attn(p);
            }
            ggml_cuda_kv_stream_cpu_attn_fold(out.data(), meta.data(), part.data(), part_meta.data(), rows);

            if (!cpu_attn_same_rows(t, "random shape matches the reference", rows, out, meta, out_ref, meta_ref)) {
                std::fprintf(stderr, "  iter=%d heads=%d/%d tokens=%d page_tokens=%u pages=%u split=%u scale=%g\n",
                    iter, n_head, n_head_kv, n_tokens, page_tokens, n_pages, split, scale);
                return;
            }
        }
    });

    t.test("cpu attention fold keeps blind rows clean", [](testing & t) {
        if (!ggml_cuda_kv_stream_cpu_attn_supported()) {
            t.skip("CPU attention needs AVX-512 F, DQ, VNNI, F16C and FMA");
            return;
        }
        constexpr int rows = 3;
        std::vector<float> acc(rows*HEAD_DIM), acc_meta(rows*2);
        std::vector<float> part(rows*HEAD_DIM), part_meta(rows*2);

        // a blind side must be skipped, not weighted by zero: 0*NaN would poison the row
        ggml_cuda_kv_stream_cpu_attn_init(acc.data(), acc_meta.data(), rows);
        ggml_cuda_kv_stream_cpu_attn_init(part.data(), part_meta.data(), rows);
        std::fill(acc.begin(), acc.end(), std::nanf(""));
        part_meta[0] = 1.5f; part_meta[1] = 2.0f;
        std::fill(part.begin(), part.begin() + HEAD_DIM, 4.0f);
        ggml_cuda_kv_stream_cpu_attn_fold(acc.data(), acc_meta.data(), part.data(), part_meta.data(), rows);
        t.assert_true("blind accumulator does not poison the part",
            acc_meta[0] == 1.5f && acc_meta[1] == 2.0f && acc[0] == 4.0f && std::isfinite(acc[HEAD_DIM - 1]));

        // a CUDA part that saw nothing reports -FLT_MAX/2 and zeros, not -inf
        ggml_cuda_kv_stream_cpu_attn_init(acc.data(), acc_meta.data(), rows);
        acc_meta[0] = 3.0f; acc_meta[1] = 5.0f;
        std::fill(acc.begin(), acc.begin() + HEAD_DIM, 7.0f);
        ggml_cuda_kv_stream_cpu_attn_init(part.data(), part_meta.data(), rows);
        part_meta[0] = -FLT_MAX/2.0f; part_meta[1] = 0.0f;
        ggml_cuda_kv_stream_cpu_attn_fold(acc.data(), acc_meta.data(), part.data(), part_meta.data(), rows);
        t.assert_true("an empty CUDA part leaves a live row unchanged",
            acc_meta[0] == 3.0f && acc_meta[1] == 5.0f && acc[0] == 7.0f);
    });

    t.test("cpu attention kernel agrees with CUDA attention", [](testing & t) {
        if (!ggml_cuda_kv_stream_cpu_attn_supported()) {
            t.skip("CPU attention needs AVX-512 F, DQ, VNNI, F16C and FMA");
            return;
        }
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        constexpr int64_t n_kv = 4*PAGE_TOKENS;
        constexpr int64_t n_batch = 4;
        constexpr uint32_t attended_pages = 3;

        cpu_attn_case c(n_kv, n_batch, {0, 1, 2});
        // run_attention always writes one KV row; hide the page holding it from both sides
        for (int64_t token = 0; token < n_batch; ++token) {
            for (int64_t cell = attended_pages*PAGE_TOKENS; cell < n_kv; ++cell) {
                c.inputs.mask[token*n_kv + cell] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
        c.set_cache_wide_mask(n_kv);

        const int64_t written_row = n_kv - 1;
        const std::vector<float> expected = run_attention(
            backend.get(), c.inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 1, 1, false, GGML_TYPE_I32, true, nullptr, false, false, 0, &written_row);

        const int rows = c.rows();
        ggml_cuda_kv_stream_cpu_attn_init(c.out.data(), c.meta.data(), rows);
        ggml_cuda_kv_stream_cpu_attn(c.make(c.out.data(), c.meta.data(), 0, attended_pages));

        if (!t.assert_equal(expected.size(), c.out.size())) {
            return;
        }
        float max_abs = 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float S = c.meta[2*r + 1];
            for (int64_t d = 0; d < HEAD_DIM; ++d) {
                const size_t i = size_t(r)*HEAD_DIM + d;
                max_abs = std::max(max_abs, std::abs(c.out[i]/S - expected[i]));
            }
        }
        // each side quantizes Q its own way, so this is close but not bit-identical
        if (!std::isfinite(max_abs) || max_abs > 5e-4f) {
            std::fprintf(stderr, "cpu vs CUDA attention max_abs=%g\n", max_abs);
        }
        t.assert_true("cpu kernel reproduces CUDA attention",
            std::isfinite(max_abs) && max_abs <= 5e-4f);
    });

    t.test("all native CUDA KV pairs preserve streamed prefill results", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 4;
        const ggml_type native_types[] = {
            GGML_TYPE_F16,
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0,
            GGML_TYPE_BF16,
        };

        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        for (const ggml_type type_k : native_types) {
            for (const ggml_type type_v : native_types) {
                const attention_inputs inputs =
                    make_inputs(n_kv, n_batch, n_kv - n_batch, type_k, type_v);
                const std::vector<float> expected = run_attention(
                    backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
                    n_kv, n_batch);

                const size_t page_bytes = query_page_bytes(backend.get(), type_k, type_v);
                ggml_backend_cuda_kv_stream_params params{};
                params.device      = 0;
                params.stage_bytes = page_bytes;
                params.stage_slots = 1;
                auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
                if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
                    return;
                }

                const std::vector<float> actual = run_attention(
                    backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
                    n_kv, n_batch);
                const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
                ggml_backend_cuda_kv_stream_runtime_free(runtime);

                if (!t.assert_equal(expected.size(), actual.size())) {
                    return;
                }
                const float max_abs = max_abs_error(expected, actual);
                if (!std::isfinite(max_abs) || max_abs > 5e-4f ||
                        stats.asynchronous_page_uploads == 0) {
                    std::fprintf(stderr,
                        "native pair K=%s V=%s max_abs=%g async_uploads=%llu\n",
                        ggml_type_name(type_k), ggml_type_name(type_v), max_abs,
                        (unsigned long long) stats.asynchronous_page_uploads);
                }
                t.assert_true("native pair executes streamed attention",
                    stats.asynchronous_page_uploads > 0);
                t.assert_true("native pair remains numerically equivalent",
                    std::isfinite(max_abs) && max_abs <= 5e-4f);
            }
        }
    });

    t.test("all bounded-fallback KV pairs preserve streamed prefill results", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 4;
        const ggml_type kv_types[] = {
            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
        };

        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        for (const ggml_type type_k : kv_types) {
            for (const ggml_type type_v : kv_types) {
                if (ggml_backend_cuda_kv_stream_get_attention_mode(type_k, type_v) !=
                        GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16) {
                    continue;
                }

                const attention_inputs inputs =
                    make_inputs(n_kv, n_batch, n_kv - n_batch, type_k, type_v);
                const attention_inputs reference = make_f16_reference(inputs, n_kv);
                const std::vector<float> expected = run_attention(
                    backend.get(), reference, ggml_backend_get_default_buffer_type(backend.get()),
                    n_kv, n_batch);

                const size_t page_bytes = query_page_bytes(backend.get(), type_k, type_v);
                const size_t conversion_bytes = query_conversion_bytes(backend.get(), type_k, type_v);

                ggml_backend_cuda_kv_stream_params params{};
                params.device           = 0;
                params.stage_bytes      = page_bytes;
                params.stage_slots      = 1;
                params.conversion_bytes = conversion_bytes;
                auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
                if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
                    return;
                }

                const std::vector<float> actual = run_attention(
                    backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
                    n_kv, n_batch);
                const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
                ggml_backend_cuda_kv_stream_runtime_free(runtime);

                if (!t.assert_equal(expected.size(), actual.size())) {
                    return;
                }
                const float max_abs = max_abs_error(expected, actual);
                if (!std::isfinite(max_abs) || max_abs > 2e-3f ||
                        stats.asynchronous_page_uploads == 0) {
                    std::fprintf(stderr,
                        "fallback pair K=%s V=%s max_abs=%g async_uploads=%llu\n",
                        ggml_type_name(type_k), ggml_type_name(type_v), max_abs,
                        (unsigned long long) stats.asynchronous_page_uploads);
                }
                t.assert_true("fallback pair executes streamed attention",
                    stats.asynchronous_page_uploads > 0);
                t.assert_true("fallback pair remains numerically equivalent",
                    std::isfinite(max_abs) && max_abs <= 2e-3f);
            }
        }
    });

    t.test("wide generic fallback remains equivalent across query workspace tiles", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 513;
        constexpr ggml_type type_k = GGML_TYPE_IQ4_NL;
        constexpr ggml_type type_v = GGML_TYPE_IQ4_NL;

        t.assert_equal(
            GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16,
            ggml_backend_cuda_kv_stream_get_attention_mode(type_k, type_v));
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs =
            make_inputs(n_kv, n_batch, n_kv - n_batch, type_k, type_v);
        const attention_inputs reference = make_f16_reference(inputs, n_kv);
        const std::vector<float> expected = run_attention(
            backend.get(), reference, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), type_k, type_v);
        const size_t conversion_bytes = query_conversion_bytes(backend.get(), type_k, type_v);

        ggml_backend_cuda_kv_stream_params params{};
        params.device           = 0;
        params.stage_bytes      = page_bytes;
        params.stage_slots      = 2;
        params.conversion_bytes = conversion_bytes;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr,
            "wide-fallback n_batch=%lld max_abs=%g async_uploads=%llu\n",
            (long long) n_batch, max_abs,
            (unsigned long long) stats.asynchronous_page_uploads);
        t.assert_true("wide fallback executes streamed attention",
            stats.asynchronous_page_uploads > 0);
        t.assert_equal(uint64_t(8), stats.host_to_device_copy_commands);
        t.assert_true("wide fallback remains numerically equivalent",
            std::isfinite(max_abs) && max_abs <= 2e-3f);
    });

    t.test("server-shaped causal prefill pipelines four Q8/Q4 blocks through two slots", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 83;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        // exercise the final causal block: with query_start == 0 every later block is masked
        const attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - 256);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch, 2, 256, true);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        std::vector<uint8_t> cleared(params.stage_bytes, 0);
        GGML_ASSERT(ggml_backend_cuda_kv_stream_stage_upload(
            runtime, 0, 0, cleared.data(), cleared.size()));

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch, 2, 256, true);


        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        t.assert_true("multi-token streamed spans use MMA partial attention",
            stats.mma_prefill_attention_spans > 0);
        t.assert_equal(uint64_t(6), stats.asynchronous_page_uploads);
        t.assert_equal(uint64_t(16), stats.host_to_device_copy_commands);
        t.assert_equal(uint64_t(6), stats.compute_stream_waits);
        t.assert_equal(uint64_t(4), stats.stage_slot_reuses);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);
        t.assert_equal(uint64_t(4), stats.streamed_attention_spans);
        t.assert_equal(uint64_t(6), stats.streamed_pages_attended);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }

        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
            max_rel = std::max(max_rel, std::abs(expected[i] - actual[i])/(std::abs(expected[i]) + 1e-6f));
        }
        std::fprintf(stderr, "streamed attention max_abs=%g max_rel=%g\n", max_abs, max_rel);
        t.assert_true("outputs remain finite", std::isfinite(max_abs) && std::isfinite(max_rel));
        t.assert_true("streamed output is numerically equivalent", max_abs <= 3e-4f);
    });

    t.test("one-page causal prefill stays bit-identical to ordinary CUDA attention", [](testing & t) {
        constexpr int64_t n_kv = 256;
        constexpr int64_t n_batch = 5;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = 0;
        params.stage_bytes = page_bytes;
        params.stage_slots = 1;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(0), stats.staged_set_rows);
        t.assert_equal(expected.size(), actual.size());
        t.assert_true("one-page outputs are bit-identical", expected == actual);
    });

    t.test("fully resident multi-page prefill stays bit-identical to ordinary CUDA attention", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 4;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch, 340);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 1;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(0), stats.streamed_pages);
        t.assert_equal(expected.size(), actual.size());
        t.assert_true("fully resident outputs are bit-identical", expected == actual);
    });

    t.test("four-query page-boundary prefill remains finite and equivalent", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 4;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch, 340);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const bool all_finite = is_finite(actual);
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "four-query page-boundary max_abs=%g\n", max_abs);
        t.assert_true("page-boundary output remains finite", all_finite);
        t.assert_true("page-boundary output remains equivalent", max_abs <= 3e-4f);
    });

    t.test("a fully masked streamed page keeps decode output equivalent", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - 1);
        for (int64_t batch = 0; batch < n_batch; ++batch) {
            for (int64_t token = 2*256; token < 3*256; ++token) {
                inputs.mask[batch*n_kv + token] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_true("fully masked page exercises streamed pages", stats.streamed_pages > 0);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const bool all_finite = is_finite(actual);
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "fully masked page max_abs=%g\n", max_abs);
        t.assert_true("fully masked page output remains finite", all_finite);
        t.assert_true("fully masked page output remains equivalent", max_abs <= 3e-4f);
    });

    t.test("non-live streamed pages are skipped by the live-page bitmap", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        struct test_case {
            uint8_t live_pages[4];
            size_t pool_pages;
            uint64_t expected_streamed;
            uint64_t expected_resident_attended;
            uint64_t expected_skipped;
        };

        const test_case cases[] = {
            { { 1, 1, 0, 1 }, 3, 2, 1, 1 },
            { { 1, 0, 0, 1 }, 3, 1, 1, 2 },
            // dead streamed tail
            { { 1, 0, 0, 0 }, 3, 0, 1, 3 },
            // trim resident head
            { { 0, 1, 1, 1 }, 4, 2, 1, 1 },
            // no live resident pages
            { { 0, 0, 1, 1 }, 4, 2, 0, 2 },
            // resident fast path
            { { 1, 1, 0, 0 }, 4, 0, 2, 2 },
            // interior dead resident page
            { { 1, 0, 1, 1 }, 5, 1, 3, 0 },
            // resident fast path starting above page 0
            { { 0, 1, 1, 1 }, 6, 0, 3, 1 },
        };

        for (const auto & tc : cases) {
            for (const int64_t n_batch : { int64_t(1), int64_t(4) }) {
                attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - 1);
                for (int64_t batch = 0; batch < n_batch; ++batch) {
                    for (int64_t token = 0; token < n_kv; ++token) {
                        if (!tc.live_pages[token/256]) {
                            inputs.mask[batch*n_kv + token] = ggml_fp32_to_fp16(-INFINITY);
                        }
                    }
                }
                const std::vector<float> expected = run_attention(
                    backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

                ggml_backend_cuda_kv_stream_params params{};
                params.device               = 0;
                params.stage_bytes          = page_bytes;
                params.stage_slots          = 2;
                params.pool_bytes           = tc.pool_pages*page_bytes;
                params.resident_layer_count = 1;
                params.page_tokens          = PAGE_TOKENS;

                auto baseline = ggml_backend_cuda_kv_stream_runtime_new(params);
                if (!t.assert_true("baseline runtime initializes", baseline != nullptr)) {
                    return;
                }
                (void) run_attention(
                    backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(baseline), n_kv, n_batch);
                const auto baseline_stats = ggml_backend_cuda_kv_stream_get_stats(baseline);
                ggml_backend_cuda_kv_stream_runtime_free(baseline);
                const uint64_t resident_pages = tc.pool_pages - params.stage_slots;
                t.assert_equal(uint64_t(4) - resident_pages, baseline_stats.streamed_pages);
                t.assert_equal(resident_pages, baseline_stats.resident_pages_attended);
                t.assert_equal(uint64_t(0), baseline_stats.skipped_pages);

                auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
                if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
                    return;
                }
                t.assert_true("live pages accepted",
                    ggml_backend_cuda_kv_stream_set_live_pages(runtime, tc.live_pages, 4));
                const std::vector<float> actual = run_attention(
                    backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
                const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
                ggml_backend_cuda_kv_stream_runtime_free(runtime);

                t.assert_equal(tc.expected_streamed, stats.streamed_pages);
                t.assert_equal(tc.expected_skipped, stats.skipped_pages);
                t.assert_equal(tc.expected_resident_attended, stats.resident_pages_attended);
                if (!t.assert_equal(expected.size(), actual.size())) {
                    return;
                }
                const bool all_finite = is_finite(actual);
                const float max_abs = max_abs_error(expected, actual);
                std::fprintf(stderr, "live-page skip n_batch=%lld max_abs=%g\n", (long long) n_batch, max_abs);
                t.assert_true("live-page skip output remains finite", all_finite);
                t.assert_true("live-page skip output remains equivalent", max_abs <= 3e-4f);
            }
        }
    });

    t.test("disjoint per-row pages keep multi-query output equivalent", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 4;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - 1);
        for (int64_t batch = 0; batch < n_batch; ++batch) {
            for (int64_t token = 0; token < n_kv; ++token) {
                if (token/256 != batch) {
                    inputs.mask[batch*n_kv + token] = ggml_fp32_to_fp16(-INFINITY);
                }
            }
        }
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_true("disjoint row test exercises streamed pages", stats.streamed_pages > 0);
        // decode-shaped batches take the vector partial, not MMA prefill
        t.assert_equal(uint64_t(0), stats.mma_prefill_attention_spans);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const bool all_finite = is_finite(actual);
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "disjoint per-row pages max_abs=%g\n", max_abs);
        t.assert_true("disjoint row output remains finite", all_finite);
        t.assert_true("disjoint row output remains equivalent", max_abs <= 3e-4f);
    });

    t.test("mixed query depths keep output equivalent across residency order", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 4;
        const int64_t depths[n_batch] = { 900, 300, 1023, 300 };
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - 1);
        for (int64_t batch = 0; batch < n_batch; ++batch) {
            for (int64_t token = depths[batch] + 1; token < n_kv; ++token) {
                inputs.mask[batch*n_kv + token] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_true("mixed depth test exercises streamed pages", stats.streamed_pages > 0);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const bool all_finite = is_finite(actual);
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "mixed query depths max_abs=%g\n", max_abs);
        t.assert_true("mixed depth output remains finite", all_finite);
        t.assert_true("mixed depth output remains equivalent", max_abs <= 3e-4f);
    });

    t.test("wide causal prefills remain equivalent across the 256-query boundary", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        const int64_t query_counts[] = { 257, 512, 513, 1024 };
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);

        for (const int64_t n_batch : query_counts) {
            const attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - n_batch);
            const std::vector<float> expected = run_attention(
                backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
                n_kv, n_batch, 1, n_batch);

            ggml_backend_cuda_kv_stream_params params{};
            params.device               = 0;
            params.stage_bytes          = page_bytes;
            params.stage_slots          = 2;
            params.pool_bytes           = 3*page_bytes;
            params.resident_layer_count = 1;
            params.page_tokens          = PAGE_TOKENS;
            auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
            if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
                return;
            }

            const std::vector<float> actual = run_attention(
                backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
                n_kv, n_batch, 1, n_batch, false, GGML_TYPE_I64, false, runtime);
            const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
            ggml_backend_cuda_kv_stream_runtime_free(runtime);

            if (!t.assert_equal(expected.size(), actual.size())) {
                return;
            }
            const bool all_finite = is_finite(actual);
            const float max_abs = max_abs_error(expected, actual);
            std::fprintf(stderr,
                "wide-query n_batch=%lld max_abs=%g streamed_pages=%llu\n",
                (long long) n_batch, max_abs,
                (unsigned long long) stats.streamed_pages);
            t.assert_true("wide-query output remains finite", all_finite);
            t.assert_true("wide-query output remains equivalent", max_abs <= 3e-4f);
            t.assert_true("wide-query test exercises streamed pages", stats.streamed_pages > 0);
        }
    });

    t.test("resident pages survive between evaluations while the tail is refreshed", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);

        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 4;
        params.pool_bytes           = 6*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(512, 256);
        (void) run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            512, 256, 2, 256, false, GGML_TYPE_I64, false, runtime);

        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        t.assert_equal(uint64_t(2), stats.resident_misses);
        t.assert_equal(uint64_t(2), stats.resident_hits);
        t.assert_equal(uint64_t(0), stats.streamed_pages);
        t.assert_equal(uint64_t(2*page_bytes), stats.host_to_device_bytes);
        t.assert_equal(uint64_t(2), stats.resident_attention_spans);
        t.assert_equal(uint64_t(4), stats.resident_pages_attended);
        t.assert_equal(uint64_t(4), stats.staged_set_rows);
        t.assert_equal(uint64_t(2*page_bytes), stats.staged_set_rows_bytes);

        t.assert_true("one resident page is demoted into the ring",
            ggml_backend_cuda_kv_stream_repartition(runtime, 5));
        t.assert_equal(uint32_t(1),
            ggml_backend_cuda_kv_stream_resident_pages_per_layer(runtime));
        (void) run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), 512, 256);

        const auto repartitioned = ggml_backend_cuda_kv_stream_get_stats(runtime);
        t.assert_equal(uint64_t(1), repartitioned.resident_misses);
        t.assert_equal(uint64_t(1), repartitioned.streamed_pages);
        t.assert_equal(uint64_t(1), repartitioned.asynchronous_page_uploads);
        t.assert_equal(uint64_t(1), repartitioned.resident_attention_spans);
        t.assert_equal(uint64_t(1), repartitioned.resident_pages_attended);

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    t.test("one-row decode refreshes only the changed resident K and V rows", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 2, 1, true, GGML_TYPE_I32, true);

        const size_t k_token_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD;
        const size_t v_token_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD;
        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 1;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 2, 1, true, GGML_TYPE_I32, true, runtime);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(2), stats.resident_misses);
        t.assert_equal(uint64_t(2), stats.resident_hits);
        t.assert_equal(
            uint64_t(2*page_bytes + k_token_bytes + v_token_bytes),
            stats.host_to_device_bytes);
        t.assert_equal(expected.size(), actual.size());
        t.assert_true("one-row resident refresh remains bit-identical", expected == actual);
    });

    t.test("resident SET_ROWS mirrors changing decode slots without page refreshes", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 5, 1, true, GGML_TYPE_I64, false, nullptr, true);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 1;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 5, 1, true, GGML_TYPE_I64, false, runtime, true);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);

        using feedback_fn_t = bool (*)(
            void *, uint64_t *, uint64_t *, double *, uint32_t *,
            uint32_t *, uint32_t *, uint32_t *, uint64_t *, uint64_t *);
        ggml_backend_dev_t device = ggml_backend_get_device(backend.get());
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto feedback_fn = reinterpret_cast<feedback_fn_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_feedback"));
        uint64_t deadline_samples = 0;
        uint64_t deadline_misses = 0;
        double copy_busy_ratio = -1.0;
        uint32_t peak_occupancy = 0;
        uint32_t ring_slots = 0;
        uint32_t resident_pages = 0;
        uint32_t controlled_pages = 0;
        uint64_t skipped_pages = 0;
        uint64_t resident_pages_attended = 0;
        t.assert_true("feedback remains readable after resident graph replay",
            feedback_fn != nullptr && feedback_fn(
                runtime, &deadline_samples, &deadline_misses, &copy_busy_ratio,
                &peak_occupancy, &ring_slots, &resident_pages, &controlled_pages,
                &skipped_pages, &resident_pages_attended));
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(2*page_bytes), stats.host_to_device_bytes);
        t.assert_equal(expected.size(), actual.size());
        t.assert_true("mirrored changing slots remain bit-identical", expected == actual);
    });

    t.test("resident staged writes support every exposed KV-cache format", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 1;
        const ggml_type kv_types[] = {
            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
        };
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        for (const ggml_type type : kv_types) {
            const attention_inputs inputs =
                make_inputs(n_kv, n_batch, n_kv - 1, type, type);
            const size_t page_bytes = query_page_bytes(backend.get(), type, type);
            const size_t conversion_bytes = query_conversion_bytes(backend.get(), type, type);

            ggml_backend_cuda_kv_stream_params baseline_params{};
            baseline_params.device           = 0;
            baseline_params.stage_bytes      = page_bytes;
            baseline_params.stage_slots      = 1;
            baseline_params.conversion_bytes = conversion_bytes;
            auto baseline_runtime = ggml_backend_cuda_kv_stream_runtime_new(baseline_params);
            if (!t.assert_true("baseline runtime initializes", baseline_runtime != nullptr)) {
                return;
            }
            const std::vector<float> expected = run_attention(
                backend.get(), inputs,
                ggml_backend_cuda_kv_stream_buffer_type(baseline_runtime),
                n_kv, n_batch, 5, 3, true, GGML_TYPE_I64, false, nullptr, true);
            ggml_backend_cuda_kv_stream_runtime_free(baseline_runtime);

            ggml_backend_cuda_kv_stream_params resident_params{};
            resident_params.device               = 0;
            resident_params.stage_bytes          = page_bytes;
            resident_params.stage_slots          = 1;
            resident_params.pool_bytes           = 3*page_bytes + conversion_bytes;
            resident_params.conversion_bytes     = conversion_bytes;
            resident_params.resident_layer_count = 1;
            resident_params.page_tokens          = PAGE_TOKENS;
            auto resident_runtime = ggml_backend_cuda_kv_stream_runtime_new(resident_params);
            if (!t.assert_true("resident runtime initializes", resident_runtime != nullptr)) {
                return;
            }
            const std::vector<float> actual = run_attention(
                backend.get(), inputs,
                ggml_backend_cuda_kv_stream_buffer_type(resident_runtime),
                n_kv, n_batch, 5, 3, true, GGML_TYPE_I64, false,
                resident_runtime, true);
            const auto stats = ggml_backend_cuda_kv_stream_get_stats(resident_runtime);
            ggml_backend_cuda_kv_stream_runtime_free(resident_runtime);

            if (!t.assert_equal(expected.size(), actual.size())) {
                return;
            }
            const float max_abs = max_abs_error(expected, actual);
            if (!t.assert_true(ggml_type_name(type),
                    std::isfinite(max_abs) && max_abs <= 5e-4f)) {
                std::fprintf(stderr, "resident staged write type=%s max_abs=%g\n",
                    ggml_type_name(type), max_abs);
                return;
            }
            t.assert_equal(uint64_t(2*page_bytes), stats.host_to_device_bytes);
            t.assert_equal(uint64_t(10), stats.staged_set_rows);
            t.assert_equal(
                uint64_t(10*3*ggml_row_size(type, HEAD_DIM*N_KV_HEAD)),
                stats.staged_set_rows_bytes);
        }
    });

    t.test("resident graph reloads after authoritative cache replacement", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 5, 1, true, GGML_TYPE_I64, false, nullptr, true, true, 1);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 1;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 5, 1, true, GGML_TYPE_I64, false, runtime, true, true, 1);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(2), stats.resident_misses);
        t.assert_equal(expected.size(), actual.size());
        t.assert_true("reloaded resident output is bit-identical", expected == actual);
    });

    t.test("decode batches contiguous streamed pages without changing logits", [](testing & t) {
        constexpr int64_t n_kv = 1536;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 2);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 4;
        params.pool_bytes           = 5*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("batched runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 2);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(10), stats.asynchronous_page_uploads);
        t.assert_equal(uint64_t(12), stats.host_to_device_copy_commands);
        t.assert_equal(uint64_t(4), stats.compute_stream_waits);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const float max_abs = max_abs_error(expected, actual);
        t.assert_true("batched streamed logits remain equivalent", max_abs <= 3e-4f);
    });

    t.test("decode pipelines bounded attention chunks and samples every immutable copy batch", [](testing & t) {
        constexpr int64_t n_kv = 41*256;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 40;
        params.pool_bytes           = 41*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        params.decode_span_pages    = 32;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("chunk-pipelined runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(40), stats.streamed_pages);
        t.assert_equal(uint64_t(2), stats.streamed_attention_spans);
        // 39 immutable pages form one 32-page and one seven-page transfer; the mutable tail
        // is excluded from adaptive prefetch feedback because its producer runs in this graph
        t.assert_equal(uint64_t(2), stats.deadline_samples);
        t.assert_true("chunk deadline misses cannot exceed samples",
            stats.deadline_misses <= stats.deadline_samples);
        t.assert_equal(uint64_t(0), stats.cpu_pages);
        t.assert_equal(uint64_t(41), stats.resident_pages_attended + stats.skipped_pages +
            stats.streamed_pages_attended + stats.cpu_pages);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const float max_abs = max_abs_error(expected, actual);
        t.assert_true("chunk-pipelined logits remain equivalent", max_abs <= 3e-4f);
    });

    t.test("layer identity survives a prefill ubatch split across graphs", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 64;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        std::vector<attention_inputs> inputs;
        for (int layer = 0; layer < 4; ++layer) {
            inputs.push_back(make_inputs(n_kv, n_batch, n_kv - 256, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 0.7f*layer));
        }
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 1, 2, GGML_TYPE_I64);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 8;
        params.pool_bytes           = 24*page_bytes;
        params.resident_layer_count = 4;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 1, 2, GGML_TYPE_I64, runtime, 0, nullptr, /* graph_per_layer = */ true);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        // the pool holds every page, so each layer attends four resident pages and streams none
        t.assert_equal(uint64_t(8), stats.staged_set_rows);
        t.assert_equal(
            uint64_t(4*(2*ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD) +
                        2*ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD))),
            stats.staged_set_rows_bytes);
        t.assert_equal(uint64_t(4), stats.resident_attention_spans);
        t.assert_equal(uint64_t(16), stats.resident_pages_attended);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const size_t per_layer = expected.size()/inputs.size();
        for (size_t layer = 0; layer < inputs.size(); ++layer) {
            float max_abs = 0.0f;
            for (size_t i = layer*per_layer; i < (layer + 1)*per_layer; ++i) {
                max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
            }
            t.assert_true("layer " + std::to_string(layer) + " attends over its own pages",
                max_abs <= 3e-4f);
        }
    });

    t.test("sixteen attention layers share one resident/ring pool during causal prefill", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 83;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        std::vector<attention_inputs> inputs(16, make_inputs(n_kv, n_batch, n_kv - 256));
        for (size_t layer = 1; layer < inputs.size(); ++layer) {
            for (size_t i = 0; i < inputs[layer].q.size(); ++i) {
                inputs[layer].q[i] += 0.025f*layer*std::sin(float(i)*0.015625f);
            }
        }
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 1, 2, GGML_TYPE_I64);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 8;
        params.pool_bytes           = 24*page_bytes;
        params.resident_layer_count = 16;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 1, 2, GGML_TYPE_I64, runtime);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        t.assert_equal(uint64_t(32), stats.staged_set_rows);
        t.assert_equal(
            uint64_t(16*(2*ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD) +
                         2*ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD))),
            stats.staged_set_rows_bytes);
        t.assert_equal(uint64_t(48), stats.asynchronous_page_uploads);
        t.assert_equal(uint64_t(48), stats.compute_stream_waits);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const float max_abs = max_abs_error(expected, actual);
        t.assert_true("sixteen-layer prefill remains equivalent", max_abs <= 3e-4f);
    });

    t.test("one shared ring prefetches across attention layers", [](testing & t) {
        constexpr int64_t n_kv = 768;
        // cross-layer prefetch is a decode optimization; a one-token batch also keeps the
        // SET_ROWS producers of this graph confined to the mutable tail page
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        std::vector<attention_inputs> inputs{
            make_inputs(n_kv, n_batch),
            make_inputs(n_kv, n_batch),
            make_inputs(n_kv, n_batch),
        };
        for (size_t layer = 1; layer < inputs.size(); ++layer) {
            for (size_t i = 0; i < inputs[layer].q.size(); ++i) {
                inputs[layer].q[i] += 0.025f*layer*std::sin(float(i)*0.015625f);
            }
        }
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 5*page_bytes;
        params.resident_layer_count = 3;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch, 2);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        t.assert_equal(uint64_t(12), stats.asynchronous_page_uploads);
        t.assert_equal(uint64_t(12), stats.compute_stream_waits);
        t.assert_equal(uint64_t(4), stats.cross_layer_prefetches);
        t.assert_equal(uint64_t(6), stats.deadline_samples);
        t.assert_true("deadline misses cannot exceed samples",
            stats.deadline_misses <= stats.deadline_samples);
        t.assert_equal(uint32_t(2), stats.ring_peak_occupancy);
        t.assert_equal(uint64_t(6), stats.streamed_attention_spans);
        t.assert_equal(uint64_t(12), stats.streamed_pages_attended);

        using feedback_fn_t = bool (*)(
            void *, uint64_t *, uint64_t *, double *, uint32_t *,
            uint32_t *, uint32_t *, uint32_t *, uint64_t *, uint64_t *);
        ggml_backend_dev_t device = ggml_backend_get_device(backend.get());
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto feedback_fn = reinterpret_cast<feedback_fn_t>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_kv_stream_feedback"));
        if (!t.assert_true("dynamic feedback API is exported", feedback_fn != nullptr)) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }
        uint64_t deadline_samples = 0;
        uint64_t deadline_misses = 0;
        double copy_busy_ratio = -1.0;
        uint32_t peak_occupancy = 0;
        uint32_t ring_slots = 0;
        uint32_t resident_pages = 0;
        uint32_t controlled_pages = 0;
        uint64_t skipped_pages = 0;
        uint64_t resident_pages_attended = 0;
        t.assert_true("dynamic feedback is readable", feedback_fn(
            runtime, &deadline_samples, &deadline_misses, &copy_busy_ratio,
            &peak_occupancy, &ring_slots, &resident_pages, &controlled_pages,
            &skipped_pages, &resident_pages_attended));
        t.assert_equal(stats.skipped_pages, skipped_pages);
        t.assert_equal(stats.resident_pages_attended, resident_pages_attended);
        t.assert_equal(uint64_t(6), deadline_samples);
        t.assert_true("copy busy ratio is normalized",
            copy_busy_ratio >= 0.0 && copy_busy_ratio <= 1.0);
        t.assert_equal(uint32_t(2), peak_occupancy);
        t.assert_equal(uint32_t(2), ring_slots);
        t.assert_equal(uint32_t(1), resident_pages);
        t.assert_equal(uint32_t(5), controlled_pages);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "multi-layer streamed attention max_abs=%g\n", max_abs);
        t.assert_true("multi-layer streamed logits remain equivalent", max_abs <= 3e-4f);
    });

    t.test("decode layout permits multiple streaming waves through a small ring", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 1;
        params.pool_bytes           = 5*page_bytes;
        params.resident_layer_count = 4;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("small-ring runtime initializes", runtime != nullptr)) {
            return;
        }

        // four layers with one resident and two streamed pages each: a one-slot ring serves
        // every layer in two waves, so one split layer per ring portion would demand eight
        t.assert_true("small ring selects a valid decode layout",
            ggml_backend_cuda_kv_stream_set_decode_layout(runtime, 3));
        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    t.test("decode layout and ring boundary publish as one reconfiguration", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 8;
        params.pool_bytes           = 12*page_bytes;
        params.resident_layer_count = 4;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        t.assert_true("combined reconfiguration succeeds",
            ggml_backend_cuda_kv_stream_reconfigure(
                runtime, /* active pages per layer = */ 3, /* ring slots = */ 4));
        t.assert_equal(uint32_t(4),
            ggml_backend_cuda_kv_stream_stage_slots(runtime));
        t.assert_equal(uint32_t(2),
            ggml_backend_cuda_kv_stream_resident_pages_per_layer(runtime));

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });


    t.test("decode layout spreads an oversized streamed deficit across enough layers", [](testing & t) {
        constexpr int64_t n_kv = 768;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        std::vector<attention_inputs> inputs{
            make_inputs(n_kv, n_batch, n_kv - 1),
            make_inputs(n_kv, n_batch, n_kv - 1),
            make_inputs(n_kv, n_batch, n_kv - 1),
            make_inputs(n_kv, n_batch, n_kv - 1),
        };
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 8;
        params.pool_bytes           = 12*page_bytes;
        params.resident_layer_count = 4;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        // four layers with three active pages and one resident page leave an eight-page deficit;
        // an eight-slot ring holds it, but a layer has only three pages, so ceil(8/3) layers split
        if (!t.assert_true("decode layout respects per-layer active-page capacity",
                ggml_backend_cuda_kv_stream_set_decode_layout(runtime, 3))) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 1, 1, GGML_TYPE_I32, runtime);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(8), stats.streamed_pages);
        t.assert_equal(uint64_t(4), stats.resident_pages_attended);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "oversized-ring decode max_abs=%g\n", max_abs);
        t.assert_true("oversized-ring decode remains equivalent", max_abs <= 3e-4f);
    });

    t.test("decode layout headroom keeps fully resident layers resident", [](testing & t) {
        constexpr int64_t n_kv = 768;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const std::vector<attention_inputs> inputs(4, make_inputs(n_kv, n_batch, n_kv - 1));
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 4;
        params.pool_bytes           = 12*page_bytes;
        params.resident_layer_count = 4;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        // target 5 over 3 attended pages gives layer_pages = [1, 1, 1, 5]
        if (!t.assert_true("decode layout accepts a target above the active set",
                ggml_backend_cuda_kv_stream_set_decode_layout(runtime, 5))) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 1, 1, GGML_TYPE_I32, runtime);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(6), stats.streamed_pages);
        t.assert_equal(uint64_t(6), stats.resident_misses);
        t.assert_equal(uint64_t(0), stats.resident_hits);
        t.assert_equal(uint64_t(6), stats.resident_pages_attended);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "headroom decode max_abs=%g\n", max_abs);
        t.assert_true("headroom decode remains equivalent", max_abs <= 3e-4f);
    });

    t.test("decode layout bounds layer concentration by the transfer ring", [](testing & t) {
        constexpr int64_t n_kv = 768;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        std::vector<attention_inputs> inputs{
            make_inputs(n_kv, n_batch, n_kv - 1),
            make_inputs(n_kv, n_batch, n_kv - 1),
            make_inputs(n_kv, n_batch, n_kv - 1),
        };
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 3;
        params.pool_bytes           = 6*page_bytes;
        params.resident_layer_count = 3;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }
        const std::vector<float> uniform = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 1, 1, GGML_TYPE_I32, runtime);
        float uniform_max_abs = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            uniform_max_abs =
                std::max(uniform_max_abs, std::abs(expected[i] - uniform[i]));
        }
        std::fprintf(stderr, "uniform decode max_abs=%g\n", uniform_max_abs);
        t.assert_true("uniform decode control remains equivalent", uniform_max_abs <= 3e-4f);
        if (!t.assert_true("decode residency becomes ring-bounded",
                ggml_backend_cuda_kv_stream_set_decode_layout(runtime, 3))) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 2, 1, GGML_TYPE_I32, runtime, 4);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(6), stats.resident_misses);
        t.assert_equal(uint64_t(0), stats.resident_hits);
        t.assert_equal(uint64_t(12), stats.streamed_pages);
        t.assert_equal(uint64_t(4), stats.resident_attention_spans);
        t.assert_equal(uint64_t(6), stats.resident_pages_attended);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        float max_abs = 0.0f;
        std::vector<float> layer_max_abs(inputs.size(), 0.0f);
        const size_t layer_elements = expected.size()/inputs.size();
        for (size_t i = 0; i < expected.size(); ++i) {
            const float error = std::abs(expected[i] - actual[i]);
            max_abs = std::max(max_abs, error);
            layer_max_abs[i/layer_elements] = std::max(layer_max_abs[i/layer_elements], error);
        }
        std::fprintf(stderr, "ring-bounded decode max_abs=%g layers=%g,%g,%g\n",
            max_abs, layer_max_abs[0], layer_max_abs[1], layer_max_abs[2]);
        t.assert_true("ring-bounded decode remains equivalent", max_abs <= 3e-4f);
    });

    t.test("multi-slot concurrent attention with staged PCIe streaming", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 2;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        // slot 0 attends 0..500 in pages 0..1; slot 1 attends 512..1000 in pages 2..3
        attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - 1);
        for (int64_t token = 0; token < n_kv; ++token) {
            if (token > 500) {
                inputs.mask[0*n_kv + token] = ggml_fp32_to_fp16(-INFINITY);
            }
            if (token < 512 || token > 1000) {
                inputs.mask[1*n_kv + token] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(3), stats.streamed_pages);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const bool all_finite = is_finite(actual);
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "multi-slot staged max_abs=%g streamed_pages=%llu\n",
            max_abs, (unsigned long long) stats.streamed_pages);
        t.assert_true("multi-slot output remains finite", all_finite);
        t.assert_true("multi-slot output remains equivalent", max_abs <= 3e-4f);
    });

    t.test("interleaved prefill and decode preserve layer identity across graphs", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr size_t n_layers = 2;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        std::vector<attention_inputs> decode_inputs{
            make_inputs(n_kv, 1, n_kv - 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 0.0f),
            make_inputs(n_kv, 1, n_kv - 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1.0f),
        };
        std::vector<attention_inputs> prefill_inputs{
            make_inputs(n_kv, 2, n_kv - 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 0.0f),
            make_inputs(n_kv, 2, n_kv - 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1.0f),
        };

        for (size_t l = 0; l < n_layers; ++l) {
            decode_inputs[l].mask[0] = ggml_fp32_to_fp16(-INFINITY);
            prefill_inputs[l].mask[0] = ggml_fp32_to_fp16(-INFINITY);
            prefill_inputs[l].mask[n_kv] = ggml_fp32_to_fp16(-INFINITY);
        }

        ggml_backend_buffer_type_t reference_buft = ggml_backend_get_default_buffer_type(backend.get());
        const std::vector<float> expected_decode = run_attention_layers(
            backend.get(), decode_inputs, reference_buft, n_kv, 1);
        std::vector<std::vector<float>> expected_prefill;
        for (size_t l = 0; l < n_layers; ++l) {
            expected_prefill.push_back(run_attention(
                backend.get(), prefill_inputs[l], reference_buft, n_kv, 2));
        }

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 4*page_bytes;
        params.resident_layer_count = n_layers;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        constexpr size_t N_TENSORS = 64;
        const size_t ctx_bytes = ggml_tensor_overhead()*N_TENSORS + ggml_graph_overhead_custom(N_TENSORS, false);
        ggml_init_params iparams{ ctx_bytes, nullptr, true };
        ggml_context_ptr kv_ctx(ggml_init(iparams));
        ggml_tensor * k_storage[n_layers];
        ggml_tensor * v_storage[n_layers];
        ggml_tensor * k_perm[n_layers];
        ggml_tensor * v_perm[n_layers];

        for (size_t l = 0; l < n_layers; ++l) {
            k_storage[l] = ggml_new_tensor_2d(kv_ctx.get(), GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD, n_kv);
            v_storage[l] = ggml_new_tensor_2d(kv_ctx.get(), GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD, n_kv);
            ggml_tensor * k_cache = ggml_view_4d(
                kv_ctx.get(), k_storage[l], HEAD_DIM, N_KV_HEAD, n_kv, 1,
                ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM),
                ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD),
                ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD)*n_kv, 0);
            ggml_tensor * v_cache = ggml_view_4d(
                kv_ctx.get(), v_storage[l], HEAD_DIM, N_KV_HEAD, n_kv, 1,
                ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM),
                ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD),
                ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD)*n_kv, 0);
            k_perm[l] = ggml_permute(kv_ctx.get(), k_cache, 0, 2, 1, 3);
            v_perm[l] = ggml_permute(kv_ctx.get(), v_cache, 0, 2, 1, 3);
        }

        ggml_backend_buffer_ptr kv_buffer(
            ggml_backend_alloc_ctx_tensors_from_buft(kv_ctx.get(), ggml_backend_cuda_kv_stream_buffer_type(runtime)));
        GGML_ASSERT(kv_buffer != nullptr);

        for (size_t l = 0; l < n_layers; ++l) {
            ggml_backend_tensor_set(k_storage[l], decode_inputs[l].k.data(), 0, decode_inputs[l].k.size());
            ggml_backend_tensor_set(v_storage[l], decode_inputs[l].v.data(), 0, decode_inputs[l].v.size());
        }

        auto run_prefill_graph = [&](size_t layer) {
            ggml_context_ptr comp_ctx(ggml_init(iparams));
            ggml_tensor * q = ggml_new_tensor_4d(comp_ctx.get(), GGML_TYPE_F32, HEAD_DIM, 2, N_Q_HEAD, 1);
            ggml_tensor * mask = ggml_new_tensor_4d(comp_ctx.get(), GGML_TYPE_F16, n_kv, 2, 1, 1);
            ggml_tensor * out = ggml_flash_attn_ext(
                comp_ctx.get(), q, k_perm[layer], v_perm[layer], mask, 1.0f/std::sqrt(float(HEAD_DIM)), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

            ggml_backend_buffer_ptr comp_buf(ggml_backend_alloc_ctx_tensors(comp_ctx.get(), backend.get()));
            const attention_inputs & input = prefill_inputs[layer];
            ggml_backend_tensor_set(q, input.q.data(), 0, input.q.size()*sizeof(float));
            ggml_backend_tensor_set(mask, input.mask.data(), 0, input.mask.size()*sizeof(uint16_t));

            ggml_cgraph * graph = ggml_new_graph_custom(comp_ctx.get(), N_TENSORS, false);
            ggml_build_forward_expand(graph, out);
            GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);

            std::vector<float> result(ggml_nelements(out));
            ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
            return result;
        };

        const std::vector<float> actual_prefill_late = run_prefill_graph(1);
        const std::vector<float> actual_prefill_early = run_prefill_graph(0);

        std::vector<float> actual_decode;
        {
            ggml_context_ptr comp_ctx(ggml_init(iparams));
            ggml_tensor * q[n_layers];
            ggml_tensor * mask[n_layers];
            ggml_tensor * out[n_layers];
            for (size_t l = 0; l < n_layers; ++l) {
                q[l] = ggml_new_tensor_4d(comp_ctx.get(), GGML_TYPE_F32, HEAD_DIM, 1, N_Q_HEAD, 1);
                mask[l] = ggml_new_tensor_4d(comp_ctx.get(), GGML_TYPE_F16, n_kv, 1, 1, 1);
                out[l] = ggml_flash_attn_ext(
                    comp_ctx.get(), q[l], k_perm[l], v_perm[l], mask[l], 1.0f/std::sqrt(float(HEAD_DIM)), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(out[l], GGML_PREC_F32);
            }

            ggml_backend_buffer_ptr comp_buf(ggml_backend_alloc_ctx_tensors(comp_ctx.get(), backend.get()));
            for (size_t l = 0; l < n_layers; ++l) {
                ggml_backend_tensor_set(q[l], decode_inputs[l].q.data(), 0, decode_inputs[l].q.size()*sizeof(float));
                ggml_backend_tensor_set(mask[l], decode_inputs[l].mask.data(), 0, decode_inputs[l].mask.size()*sizeof(uint16_t));
            }

            ggml_cgraph * graph = ggml_new_graph_custom(comp_ctx.get(), N_TENSORS, false);
            for (size_t l = 0; l < n_layers; ++l) {
                ggml_build_forward_expand(graph, out[l]);
            }
            GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);

            for (size_t l = 0; l < n_layers; ++l) {
                const size_t begin = actual_decode.size();
                actual_decode.resize(begin + ggml_nelements(out[l]));
                ggml_backend_tensor_get(out[l], actual_decode.data() + begin, 0, ggml_nbytes(out[l]));
            }
        }
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        auto compare = [&t](const std::string & name,
                const std::vector<float> & expected, const std::vector<float> & actual) {
            if (!t.assert_equal(name + " output size", expected.size(), actual.size())) {
                return;
            }
            const bool all_finite = is_finite(actual);
            const float max_abs = max_abs_error(expected, actual);
            std::fprintf(stderr, "%s max_abs=%g\n", name.c_str(), max_abs);
            t.assert_true(name + " stays finite", all_finite);
            t.assert_true(name + " keeps its own layer payload", max_abs <= 3e-4f);
        };

        t.assert_equal(uint64_t(4), stats.streamed_pages);
        compare("prefill on the late layer", expected_prefill[1], actual_prefill_late);
        compare("prefill on the early layer", expected_prefill[0], actual_prefill_early);
        compare("decode across both layers", expected_decode, actual_decode);
    });

    t.test("scattered multi-slot set_rows preserves resident mirror coherence across steps", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 2;
        constexpr int64_t update_rows = 2;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - 1);
        for (int64_t token = 0; token < n_kv; ++token) {
            if (token >= 256) {
                inputs.mask[0*n_kv + token] = ggml_fp32_to_fp16(-INFINITY);
            } else {
                inputs.mask[1*n_kv + token] = ggml_fp32_to_fp16(-INFINITY);
            }
        }

        const int64_t scattered_rows[update_rows] = { 50, 300 };
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 2, update_rows, true, GGML_TYPE_I64, false, nullptr, false, false, 0, scattered_rows);

        const size_t page_bytes = query_page_bytes(backend.get(), inputs.type_k, inputs.type_v);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 4*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 2, update_rows, true, GGML_TYPE_I64, false, runtime, false, false, 0, scattered_rows);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(0), stats.streamed_pages);
        t.assert_equal(uint64_t(0), stats.staged_set_rows);
        t.assert_equal(uint64_t(2), stats.resident_hits);
        t.assert_equal(uint64_t(2), stats.resident_misses);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const bool all_finite = is_finite(actual);
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "scattered multi-slot set_rows max_abs=%g\n", max_abs);
        t.assert_true("scattered multi-slot output remains finite", all_finite);
        t.assert_true("scattered multi-slot writes keep resident mirror coherent", max_abs <= 3e-4f);
    });

    t.test("mid-span mutable pages do not stall cross-layer prefetch", [](testing & t) {
        constexpr int64_t n_kv = 8*256;
        constexpr int64_t n_batch = 1;
        constexpr int64_t update_rows = 2;
        constexpr size_t n_layers = 2;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        // pages 1, 5, 7 mutable; pages 2, 3, 4, 6 immutable streamed
        const int64_t rows[update_rows] = { 300, 1300 };
        std::vector<attention_inputs> inputs{
            make_inputs(n_kv, n_batch, n_kv - 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 0.0f),
            make_inputs(n_kv, n_batch, n_kv - 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1.0f),
        };
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 1, update_rows, GGML_TYPE_I64, nullptr, 0, rows);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 8;
        params.pool_bytes           = (n_layers + 8)*page_bytes;
        params.resident_layer_count = n_layers;
        params.page_tokens          = PAGE_TOKENS;
        params.decode_span_pages    = 32;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 1, update_rows, GGML_TYPE_I64, runtime, 0, rows);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(14), stats.streamed_pages);
        t.assert_equal(uint64_t(4), stats.cross_layer_prefetches);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "mid-span mutable max_abs=%g cross=%llu\n",
            max_abs, (unsigned long long) stats.cross_layer_prefetches);
        t.assert_true("mid-span mutable output remains equivalent", max_abs <= 3e-4f);
    });

    t.test("two-slot decode pipelines streamed pages across layers", [](testing & t) {
        constexpr int64_t n_kv = 8*256;
        constexpr int64_t n_batch = 2;
        constexpr int64_t update_rows = 2;
        constexpr size_t n_layers = 2;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        // slot 0 attends 0..300 (row 300); slot 1 attends 1024..1300 (row 1300)
        const int64_t rows[update_rows] = { 300, 1300 };
        std::vector<attention_inputs> inputs{
            make_inputs(n_kv, n_batch, n_kv - 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 0.0f),
            make_inputs(n_kv, n_batch, n_kv - 1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 0.5f),
        };
        for (auto & input : inputs) {
            for (int64_t token = 0; token < n_kv; ++token) {
                if (token > 300) {
                    input.mask[token] = ggml_fp32_to_fp16(-INFINITY);
                }
                if (token < 1024 || token > 1300) {
                    input.mask[n_kv + token] = ggml_fp32_to_fp16(-INFINITY);
                }
            }
        }
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 1, update_rows, GGML_TYPE_I64, nullptr, 0, rows);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 8;
        params.pool_bytes           = (n_layers + 8)*page_bytes;
        params.resident_layer_count = n_layers;
        params.page_tokens          = PAGE_TOKENS;
        params.decode_span_pages    = 32;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 1, update_rows, GGML_TYPE_I64, runtime, 0, rows);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(14), stats.streamed_pages);
        t.assert_true("two-slot decode is graph planned", stats.deadline_samples > 0);
        t.assert_equal(uint64_t(4), stats.cross_layer_prefetches);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const bool all_finite = is_finite(actual);
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "two-slot pipelined max_abs=%g samples=%llu cross=%llu\n",
            max_abs, (unsigned long long) stats.deadline_samples,
            (unsigned long long) stats.cross_layer_prefetches);
        t.assert_true("two-slot pipelined output remains finite", all_finite);
        t.assert_true("two-slot pipelined output remains equivalent", max_abs <= 3e-4f);
    });

    t.test("two-query decode keeps the bounded streamed span", [](testing & t) {
        constexpr int64_t n_kv = 41*256;
        constexpr int64_t n_batch = 2;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - 2);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch);

        const size_t page_bytes = query_page_bytes(backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 40;
        params.pool_bytes           = 41*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = PAGE_TOKENS;
        params.decode_span_pages    = 32;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("chunk-pipelined runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(40), stats.streamed_pages);
        t.assert_equal(uint64_t(2), stats.streamed_attention_spans);
        t.assert_equal(uint64_t(2), stats.deadline_samples);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        const float max_abs = max_abs_error(expected, actual);
        std::fprintf(stderr, "two-query bounded span max_abs=%g spans=%llu\n",
            max_abs, (unsigned long long) stats.streamed_attention_spans);
        t.assert_true("two-query bounded span output remains equivalent", max_abs <= 3e-4f);
    });

    ggml_quantize_free();
    return t.summary();
}
