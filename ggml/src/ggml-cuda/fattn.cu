#include "common.cuh"
#include "convert.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "kv-stream-cpu-attn.h"
#include "kv-stream-cpu-pages.h"
#include "kv-stream-cpu-pool.h"
#include "kv-stream-geometry.h"
#include "kv-stream-span-tuner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, const int ne30, const int n_kv_max,
        const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int query    = blockIdx.x;

    const half * mask = mask_ptr + sequence*s33 + query*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + query)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const bool selected = i < ne30 && isfinite(__half2float(mask[i]));
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = row_count;
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num(mask->ne[1], mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_mask_to_sparse_indices, launch_params,
        (const half *) mask->data, indices, int(mask->ne[0]), n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2LL*n_kv_max);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}
struct ggml_cuda_kv_stream_resident_cache;

namespace {

constexpr size_t KV_STREAM_NO_REQUEST = std::numeric_limits<size_t>::max();
constexpr uint32_t KV_STREAM_NO_LAYER = std::numeric_limits<uint32_t>::max();
constexpr uint32_t KV_STREAM_COPY_BATCH_PAGES = 32;
constexpr uint32_t KV_STREAM_DECODE_SPAN_PAGES = KV_STREAM_COPY_BATCH_PAGES;

struct kv_stream_chunk_geometry {
    int64_t token_begin = 0;
    int64_t token_count = 0;
    size_t k_row_bytes = 0;
    size_t v_row_bytes = 0;
    int64_t n_head_kv = 0;

    kv_stream_chunk_geometry() = default;
    kv_stream_chunk_geometry(const ggml_tensor * K, const ggml_tensor * V, int64_t token_begin, int64_t token_count) :
        token_begin(token_begin), token_count(token_count),
        k_row_bytes(ggml_row_size(K->type, K->ne[0])), v_row_bytes(ggml_row_size(V->type, V->ne[0])),
        n_head_kv(K->ne[2]) {
        GGML_ASSERT(V->ne[2] == n_head_kv);
        GGML_ASSERT(K->nb[1] == k_token_stride() && K->nb[2] == k_row_bytes);
        GGML_ASSERT(V->nb[1] == v_token_stride() && V->nb[2] == v_row_bytes);
    }

    size_t k_token_stride() const { return k_row_bytes*n_head_kv; }
    size_t v_token_stride() const { return v_row_bytes*n_head_kv; }
    size_t k_bytes() const { return k_token_stride()*token_count; }
    size_t v_bytes() const { return v_token_stride()*token_count; }
    size_t v_offset() const { return GGML_PAD(k_bytes(), 128); }
    size_t page_bytes() const { return v_offset() + v_bytes(); }

    bool follows(const kv_stream_chunk_geometry & previous) const {
        return k_row_bytes == previous.k_row_bytes && v_row_bytes == previous.v_row_bytes &&
            n_head_kv == previous.n_head_kv && token_count == previous.token_count &&
            token_begin == previous.token_begin + previous.token_count;
    }
};

struct kv_stream_graph_request {
    const char * k_data = nullptr;
    const char * v_data = nullptr;
    kv_stream_chunk_geometry geometry;
    uint32_t layer = 0;
    uint32_t slot = 0;
    uint32_t ready_slot = 0;
    bool mutable_tail = false;
    bool eligible = false;
    bool scheduled = false;
    bool consumed = false;
    bool deadline_sample = false;
};

struct kv_stream_layer_plan {
    uint32_t layer = 0;
    std::vector<size_t> request_by_page;
    std::vector<size_t> mutable_requests;
    std::vector<uint32_t> cpu_pages;
};

// TODO: Task B skeleton knobs; Task D keeps one body, join and part path, Task F sets the default share
struct kv_stream_cpu_knobs {
    uint32_t pages_per_layer = 0;
    float share = 0.0f;
    bool join_blocking = false;
    bool spin = false;
    double dram_bandwidth_gbps = 0.0;
    bool copy_part_to_device = false;
};

struct kv_stream_cpu_job_record;

struct kv_stream_cpu_job {
    ggml_cuda_kv_stream_cpu_attn_params params;
    std::vector<uint32_t> pages;
    mutable bool result_initialized = false;
    // TODO: skeleton fields, removed with the spin body and the counters
    bool spin = false;
    int64_t spin_ns = 0;
    kv_stream_cpu_job_record * record = nullptr;
};

using kv_stream_cpu_pool = ggml_cuda_kv_stream_cpu_pool<kv_stream_cpu_job>;

// TODO: Task B skeleton counters; drop them once Task D has re-measured the fixed cost
struct kv_stream_cpu_job_record {
    kv_stream_cpu_pool * pool = nullptr;
    uint32_t layer = 0;
    uint32_t pages = 0;
    uint32_t queued = 0;
    int64_t spin_ns = 0;
    int64_t arm_ns = 0;
    int64_t release_ns = 0;
    std::vector<int64_t> start_ns;
    std::vector<int64_t> kernel_ns;
    std::vector<int64_t> end_ns;
    std::vector<uint32_t> mask_zero;
};

struct alignas(64) kv_stream_cpu_line {
    uint8_t bytes[64];
};
static_assert(GGML_CUDA_KV_STREAM_CPU_ATTN_ROW_WSIZE % sizeof(kv_stream_cpu_line) == 0,
    "row workspace size must be a multiple of cache line size");

struct kv_stream_cpu_worker {
    std::vector<float> part;
    std::vector<float> part_meta;
    std::vector<kv_stream_cpu_line> q_workspace;

    void run(const ggml_cuda_kv_stream_cpu_attn_params & geom, const uint32_t * pages, uint32_t n_pages) {
        ggml_cuda_kv_stream_cpu_attn_params p = geom;
        p.pages = pages;
        p.n_pages = n_pages;
        p.out = part.data();
        p.out_meta = part_meta.data();
        p.wdata = q_workspace.data();
        p.wsize = q_workspace.size()*sizeof(kv_stream_cpu_line);
        ggml_cuda_kv_stream_cpu_attn(p);
    }
};

static int64_t kv_stream_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool kv_stream_wait_value(cudaStream_t stream, uint32_t * flag_device, uint32_t value) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
    return cuStreamWaitValue32(stream, reinterpret_cast<CUdeviceptr>(flag_device), value, CU_STREAM_WAIT_VALUE_GEQ) ==
        CUDA_SUCCESS;
#else
    GGML_UNUSED_VARS(stream, flag_device, value);
    return false;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
}

// TODO: Task B skeleton counters; Task D deletes this struct, the job's record pointer and their callers
struct kv_stream_cpu_diag {
    std::vector<kv_stream_cpu_job_record> records;
    uint32_t graph_jobs = 0;
    uint32_t layout_ring = 0;
    std::vector<std::array<uint32_t, 4>> layout;
    std::vector<std::array<uint32_t, 4>> printed_layout;
    std::vector<std::string> report;
};

struct kv_stream_cpu_graph {
    kv_stream_cpu_knobs knobs;
    const ggml_tensor * mask = nullptr;
    uint32_t mask_page_begin = UINT32_MAX;
    uint32_t mask_page_end = 0;
    bool mask_copied = false;
    int64_t mask_rows = 0;
};

struct kv_stream_cpu_split {
    // set once, by set_cpu_split
    size_t max_rows = 0;
    uint32_t context_pages = 0;
    cudaStream_t side_stream = nullptr;
    float * q = nullptr;
    uint16_t * mask = nullptr;
    float * result = nullptr;
    float * result_meta = nullptr;
    float * result_device = nullptr;
    float2 * result_meta_device = nullptr;
    std::atomic<uint32_t> * done = nullptr;
    uint32_t * done_device = nullptr;
    bool supports_stream_join = false;
    // execution
    std::vector<kv_stream_cpu_worker> workers;
    std::mutex fold_mutex;
    std::unique_ptr<kv_stream_cpu_pool> pool;
    // per graph
    kv_stream_cpu_graph graph;
    // TODO: Task B skeleton counters
    kv_stream_cpu_diag diag;

    uint32_t warned_blocks = 0;
    void warn_block(ggml_cuda_kv_stream_cpu_split_block block) {
        static_assert(GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_COUNT <= 32, "block reasons must fit the warned bitmask");
        const uint32_t bit = 1u << uint32_t(block);
        if ((warned_blocks & bit) == 0) {
            warned_blocks |= bit;
            GGML_LOG_WARN("kv stream cpu split disabled: %s\n", ggml_cuda_kv_stream_cpu_split_block_reason(block));
        }
    }

    ~kv_stream_cpu_split() {
        CUDA_CHECK(cudaDeviceSynchronize());
        pool.reset();
        if (side_stream != nullptr) {
            CUDA_CHECK(cudaStreamDestroy(side_stream));
        }
        CUDA_CHECK(cudaFreeHost(q));
        CUDA_CHECK(cudaFreeHost(mask));
        CUDA_CHECK(cudaFreeHost(result));
        CUDA_CHECK(cudaFreeHost(result_meta));
        CUDA_CHECK(cudaFreeHost(done));
    }
};

// TODO: O-D2's count of cpu pages every token sees unbiased, removed with the skeleton counters
static uint32_t kv_stream_cpu_mask_zero_pages(
        const ggml_cuda_kv_stream_cpu_attn_params & p, const uint32_t * pages, uint32_t n_pages) {
    uint32_t zero = 0;
    for (uint32_t i = 0; i < n_pages; ++i) {
        bool all = true;
        for (int t = 0; t < p.n_tokens && all; ++t) {
            const uint16_t * row = (const uint16_t *)((const uint8_t *) p.mask + size_t(t)*p.mask_token_stride) +
                size_t(pages[i])*p.page_tokens;
            all = std::all_of(row, row + p.page_tokens, [](uint16_t v) { return (v & 0x7fff) == 0; });
        }
        zero += all;
    }
    return zero;
}

static void kv_stream_cpu_job_run(kv_stream_cpu_split & split, const kv_stream_cpu_job & job, int thread, int n_threads) {
    kv_stream_cpu_job_record & record = *job.record;
    const int rows = job.params.n_tokens*job.params.n_head;
    const uint32_t n_pages = uint32_t(job.pages.size());
    const uint32_t begin = uint32_t(uint64_t(n_pages)*thread/n_threads);
    const uint32_t end = uint32_t(uint64_t(n_pages)*(thread + 1)/n_threads);
    // the diagnostic scan stays out of busy
    record.mask_zero[thread] = kv_stream_cpu_mask_zero_pages(job.params, job.pages.data() + begin, end - begin);
    record.start_ns[thread] = kv_stream_now_ns();

    // TODO: the spin body goes with the skeleton
    if (job.spin) {
        if (thread == 0) {
            ggml_cuda_kv_stream_cpu_attn_init(split.result, split.result_meta, rows);
        }
        const int64_t deadline = record.start_ns[thread] + job.spin_ns;
        while (kv_stream_now_ns() < deadline) {
        }
        record.kernel_ns[thread] = record.end_ns[thread] = kv_stream_now_ns();
        return;
    }

    kv_stream_cpu_worker & worker = split.workers[thread];
    ggml_cuda_kv_stream_cpu_attn_init(worker.part.data(), worker.part_meta.data(), rows);
    if (end > begin) {
        worker.run(job.params, job.pages.data() + begin, end - begin);
    }
    record.kernel_ns[thread] = kv_stream_now_ns();
    // TODO: Task D5 chooses the fold strategy
    {
        std::lock_guard<std::mutex> lock(split.fold_mutex);
        if (!job.result_initialized) {
            job.result_initialized = true;
            ggml_cuda_kv_stream_cpu_attn_init(split.result, split.result_meta, rows);
        }
        ggml_cuda_kv_stream_cpu_attn_fold(
            split.result, split.result_meta, worker.part.data(), worker.part_meta.data(), rows);
    }
    record.end_ns[thread] = kv_stream_now_ns();
}

// TODO: env-only pinning until Task F settles O-C1
// "2-7" or "1,3,5-7"; empty on a parse error
static std::vector<int> kv_stream_cpu_parse_cpus(const char * text) {
    std::vector<int> cpus;
    const char * p = text;
    while (*p != '\0') {
        char * end = nullptr;
        const long first = strtol(p, &end, 10);
        long last = first;
        if (end != p && *end == '-') {
            p = end + 1;
            last = strtol(p, &end, 10);
        }
        // 1024 is CPU_SETSIZE on glibc
        if (end == p || first < 0 || last < first || last >= 1024 || (*end != ',' && *end != '\0')) {
            return {};
        }
        for (long cpu = first; cpu <= last; ++cpu) {
            cpus.push_back(int(cpu));
        }
        p = *end == ',' ? end + 1 : end;
    }
    return cpus;
}

static std::vector<int> kv_stream_cpu_env_cpus() {
    const char * text = getenv("GGML_CUDA_KV_STREAM_CPU_CPUS");
    if (text == nullptr || *text == '\0') {
        return {};
    }
    std::vector<int> cpus = kv_stream_cpu_parse_cpus(text);
    if (cpus.empty()) {
        GGML_LOG_WARN("%s: ignoring GGML_CUDA_KV_STREAM_CPU_CPUS=%s, expected a list like 2-7 or 1,3,5-7\n",
            __func__, text);
        return {};
    }
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        GGML_LOG_WARN("%s: cannot read the process affinity, the split pool runs unpinned\n", __func__);
        return {};
    }
    for (const int cpu : cpus) {
        if (cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed)) {
            GGML_LOG_WARN("%s: cpu %d of GGML_CUDA_KV_STREAM_CPU_CPUS=%s is outside the process affinity mask, "
                "the split pool runs unpinned\n", __func__, cpu, text);
            return {};
        }
    }
    return cpus;
#else
    GGML_LOG_WARN("%s: ignoring GGML_CUDA_KV_STREAM_CPU_CPUS=%s, CPU pinning needs Linux\n", __func__, text);
    return {};
#endif
}

static bool kv_stream_cpu_env_number(const char * name, double & value) {
    const char * text = getenv(name);
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char * end = nullptr;
    value = strtod(text, &end);
    return *end == '\0';
}

static kv_stream_cpu_knobs kv_stream_cpu_read_knobs(bool stream_join) {
    const auto env = [](const char * name) {
        const char * value = getenv(name);
        return value == nullptr ? "" : value;
    };
    kv_stream_cpu_knobs knobs;
    double value = 0.0;
    if (kv_stream_cpu_env_number("GGML_CUDA_KV_STREAM_CPU_PAGES", value) && value >= 0.0 && value <= double(UINT32_MAX)) {
        knobs.pages_per_layer = uint32_t(value);
    }
    if (kv_stream_cpu_env_number("GGML_CUDA_KV_STREAM_CPU_SHARE", value) && value > 0.0) {
        knobs.share = std::min(float(value), 1.0f);
    }
    knobs.join_blocking = !stream_join || strcmp(env("GGML_CUDA_KV_STREAM_CPU_JOIN"), "block") == 0;
    knobs.spin = strcmp(env("GGML_CUDA_KV_STREAM_CPU_BODY"), "spin") == 0;
    if (kv_stream_cpu_env_number("GGML_CUDA_KV_STREAM_CPU_BCPU", value) && value > 0.0) {
        knobs.dram_bandwidth_gbps = value;
    }
    knobs.copy_part_to_device = strcmp(env("GGML_CUDA_KV_STREAM_CPU_PART"), "copy") == 0;
    return knobs;
}

} // namespace

struct ggml_cuda_kv_stream_transfer_ring {
    char * pool_data = nullptr;
    size_t page_bytes = 0;
    char * conversion_data = nullptr;
    size_t conversion_bytes = 0;
    uint32_t capacity_slots = 0;
    uint32_t active_slots = 0;
    uint32_t forced_decode_span_pages = 0;
    uint32_t graph_decode_span_pages = UINT32_MAX;
    uint32_t graph_copy_batch_pages = KV_STREAM_COPY_BATCH_PAGES;
    bool graph_copy_batch_greedy = false;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t producer_ready = nullptr;
    cudaEvent_t eval_start = nullptr;
    cudaEvent_t eval_end = nullptr;
    cudaEvent_t copy_sample_start = nullptr;
    cudaEvent_t copy_sample_end = nullptr;
    std::vector<cudaEvent_t> ready;
    std::vector<cudaEvent_t> consumed;
    std::vector<uint8_t> slot_used;
    std::vector<size_t> slot_request;
    std::vector<uint8_t> live_pages;
    uint32_t * ready_flags_host = nullptr;
    uint32_t * ready_flags_device = nullptr;
    uint64_t * deadline_counters_host = nullptr;
    uint64_t * deadline_counters_device = nullptr;

    bool graph_active = false;
    bool graph_decode = true;
    ggml_cuda_kv_stream_span_tuner span_tuner;
    uint32_t graph_layer_count = 0;
    uint32_t current_layer = KV_STREAM_NO_LAYER;
    size_t next_request = 0;
    ggml_cuda_kv_stream_resident_cache * graph_resident_cache = nullptr;
    std::vector<kv_stream_graph_request> graph_requests;
    std::unordered_map<const void *, kv_stream_layer_plan> graph_plans;
    std::unique_ptr<kv_stream_cpu_split> cpu_split;

    uint64_t asynchronous_page_uploads = 0;
    uint64_t host_to_device_copy_commands = 0;
    uint64_t compute_stream_waits = 0;
    uint64_t stage_slot_reuses = 0;
    uint64_t skipped_pages = 0;
    uint64_t cross_layer_prefetches = 0;
    uint64_t cpu_pages = 0;
    uint64_t cpu_jobs = 0;
    uint32_t current_occupancy = 0;
    uint32_t ring_peak_occupancy = 0;
    uint32_t current_ring_peak_occupancy = 0;
    uint32_t last_ring_peak_occupancy = 0;
    uint64_t current_epoch_uploads = 0;
    double last_copy_engine_busy_ratio = 0.0;
    uint32_t copy_sample_uploads = 0;
    bool timing_pending = false;
    bool timing_current = false;
    bool last_graph_decode = false;
    bool last_graph_copy_batch_greedy = false;
    bool last_graph_streamed = false;
    bool copy_sample_recorded = false;
};

ggml_cuda_kv_stream_transfer_ring * ggml_cuda_kv_stream_transfer_ring_new(
        void * pool_data, size_t page_bytes, uint32_t stage_slots,
        void * conversion_data, size_t conversion_bytes,
        uint32_t forced_decode_span_pages) {
    if (pool_data == nullptr || page_bytes == 0 || stage_slots == 0 ||
            (conversion_bytes != 0 && conversion_data == nullptr)) {
        return nullptr;
    }

    auto * ring = new ggml_cuda_kv_stream_transfer_ring;
    ring->pool_data = static_cast<char *>(pool_data);
    ring->page_bytes = page_bytes;
    ring->conversion_data = static_cast<char *>(conversion_data);
    ring->conversion_bytes = conversion_bytes;
    ring->capacity_slots = stage_slots;
    ring->active_slots = stage_slots;
    ring->ready.resize(stage_slots, nullptr);
    ring->forced_decode_span_pages = forced_decode_span_pages;
    ring->consumed.resize(stage_slots, nullptr);
    ring->slot_used.resize(stage_slots, 0);
    ring->slot_request.resize(stage_slots, KV_STREAM_NO_REQUEST);

    auto cleanup = [&]() {
        for (cudaEvent_t event : ring->ready) {
            if (event != nullptr) {
                (void) cudaEventDestroy(event);
            }
        }
        for (cudaEvent_t event : ring->consumed) {
            if (event != nullptr) {
                (void) cudaEventDestroy(event);
            }
        }
        if (ring->producer_ready != nullptr) {
            (void) cudaEventDestroy(ring->producer_ready);
        }
        if (ring->eval_start != nullptr) {
            (void) cudaEventDestroy(ring->eval_start);
        }
        if (ring->eval_end != nullptr) {
            (void) cudaEventDestroy(ring->eval_end);
        }
        if (ring->copy_sample_start != nullptr) {
            (void) cudaEventDestroy(ring->copy_sample_start);
        }
        if (ring->copy_sample_end != nullptr) {
            (void) cudaEventDestroy(ring->copy_sample_end);
        }
        if (ring->copy_stream != nullptr) {
            (void) cudaStreamDestroy(ring->copy_stream);
        }
        if (ring->ready_flags_host != nullptr) {
            (void) cudaFreeHost(ring->ready_flags_host);
        }
        if (ring->deadline_counters_host != nullptr) {
            (void) cudaFreeHost(ring->deadline_counters_host);
        }
        delete ring;
    };

    if (cudaHostAlloc(reinterpret_cast<void **>(&ring->ready_flags_host),
            stage_slots*sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&ring->ready_flags_device),
            ring->ready_flags_host, 0) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&ring->deadline_counters_host),
            2*sizeof(uint64_t), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&ring->deadline_counters_device),
            ring->deadline_counters_host, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        cleanup();
        return nullptr;
    }
    std::fill_n(ring->ready_flags_host, stage_slots, 0u);
    std::fill_n(ring->deadline_counters_host, 2, uint64_t(0));

    if (cudaStreamCreateWithFlags(&ring->copy_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&ring->producer_ready, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreate(&ring->eval_start) != cudaSuccess ||
        cudaEventCreate(&ring->eval_end) != cudaSuccess ||
        cudaEventCreate(&ring->copy_sample_start) != cudaSuccess ||
        cudaEventCreate(&ring->copy_sample_end) != cudaSuccess) {
        (void) cudaGetLastError();
        cleanup();
        return nullptr;
    }
    for (uint32_t slot = 0; slot < stage_slots; ++slot) {
        if (cudaEventCreateWithFlags(&ring->ready[slot], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ring->consumed[slot], cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            cleanup();
            return nullptr;
        }
    }
    return ring;
}

void ggml_cuda_kv_stream_transfer_ring_free(ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(ring->copy_stream));
    for (cudaEvent_t event : ring->ready) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    for (cudaEvent_t event : ring->consumed) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    CUDA_CHECK(cudaEventDestroy(ring->producer_ready));
    CUDA_CHECK(cudaEventDestroy(ring->eval_start));
    CUDA_CHECK(cudaEventDestroy(ring->eval_end));
    CUDA_CHECK(cudaEventDestroy(ring->copy_sample_start));
    CUDA_CHECK(cudaEventDestroy(ring->copy_sample_end));
    CUDA_CHECK(cudaStreamDestroy(ring->copy_stream));
    CUDA_CHECK(cudaFreeHost(ring->ready_flags_host));
    CUDA_CHECK(cudaFreeHost(ring->deadline_counters_host));
    delete ring;
}

bool ggml_cuda_kv_stream_transfer_ring_set_active_slots(
        ggml_cuda_kv_stream_transfer_ring * ring, uint32_t stage_slots) {
    if (ring == nullptr || stage_slots == 0 || stage_slots > ring->capacity_slots) {
        return false;
    }
    if (ring->active_slots != stage_slots) {
        ring->span_tuner.reset();
        ring->timing_pending = false;
        ring->timing_current = false;
    }
    ring->active_slots = stage_slots;
    return true;
}

void ggml_cuda_kv_stream_transfer_ring_set_live_pages(
        ggml_cuda_kv_stream_transfer_ring * ring, const uint8_t * live_pages, size_t count) {
    if (ring == nullptr) {
        return;
    }
    if (live_pages == nullptr || count == 0 ||
            std::none_of(live_pages, live_pages + count, [](uint8_t v) { return v != 0; })) {
        ring->live_pages.clear();
        return;
    }
    ring->live_pages.assign(live_pages, live_pages + count);
}

bool ggml_cuda_kv_stream_transfer_ring_set_cpu_split(
        ggml_cuda_kv_stream_transfer_ring * ring, uint32_t n_threads, uint32_t n_head, uint32_t context_pages,
        uint32_t n_layers) {
    if (ring == nullptr || n_threads == 0 || n_head == 0 || context_pages == 0 || n_layers == 0) {
        return false;
    }
    GGML_ASSERT(ggml_cuda_kv_stream_cpu_attn_supported());
    GGML_ASSERT(ring->cpu_split == nullptr);

    const size_t rows = size_t(GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS)*n_head;
    const size_t row_bytes = rows*GGML_CUDA_KV_STREAM_HEAD_DIM*sizeof(float);
    const size_t mask_bytes = size_t(GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS)*context_pages*
        GGML_CUDA_KV_STREAM_PAGE_TOKENS*sizeof(uint16_t);
    const size_t meta_bytes = rows*2*sizeof(float);

    static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
        "the GPU reads the job flag as a plain uint32_t");

    auto split = std::make_unique<kv_stream_cpu_split>();
    split->max_rows = rows;
    split->context_pages = context_pages;
    if (cudaStreamCreateWithFlags(&split->side_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&split->q), row_bytes, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&split->mask), mask_bytes, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&split->result), row_bytes, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&split->result_meta), meta_bytes, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&split->done), sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&split->result_device), split->result, 0) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&split->result_meta_device), split->result_meta, 0) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&split->done_device), split->done, 0) != cudaSuccess) {
        const cudaError_t error = cudaGetLastError();
        GGML_LOG_WARN("%s: allocating %.2f MiB of pinned CPU split scratch failed, streamed pages stay on the GPU: %s\n",
            __func__, (2*row_bytes + mask_bytes + meta_bytes)/1024.0/1024.0, cudaGetErrorString(error));
        return false;
    }
    // never reset
    new (split->done) std::atomic<uint32_t>(0);
    split->supports_stream_join = kv_stream_wait_value(split->side_stream, split->done_device, 0);
    if (!split->supports_stream_join) {
        GGML_LOG_WARN("%s: this build or device cannot wait on a host flag, CPU split jobs join on the submit thread\n",
            __func__);
    }
    ggml_cuda_kv_stream_cpu_attn_init(split->result, split->result_meta, int(rows));
    split->workers.resize(n_threads);
    for (kv_stream_cpu_worker & worker : split->workers) {
        worker.part.resize(rows*GGML_CUDA_KV_STREAM_HEAD_DIM);
        worker.part_meta.resize(rows*2);
        worker.q_workspace.resize(rows*GGML_CUDA_KV_STREAM_CPU_ATTN_ROW_WSIZE/sizeof(kv_stream_cpu_line));
    }
    const std::vector<int> cpus = kv_stream_cpu_env_cpus();
    kv_stream_cpu_split * body_split = split.get();
    split->pool = std::make_unique<kv_stream_cpu_pool>(int(n_threads), n_layers,
        [body_split](const kv_stream_cpu_job & job, int thread, int n) {
            kv_stream_cpu_job_run(*body_split, job, thread, n);
        }, split->done, cpus);
    split->diag.records.resize(n_layers);
    for (kv_stream_cpu_job_record & record : split->diag.records) {
        record.pool = split->pool.get();
        record.start_ns.resize(n_threads);
        record.kernel_ns.resize(n_threads);
        record.end_ns.resize(n_threads);
        record.mask_zero.resize(n_threads);
    }
    ring->cpu_split = std::move(split);
    return true;
}

void ggml_cuda_kv_stream_transfer_ring_reset_span_tuner(
        ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring != nullptr) {
        ring->span_tuner.reset();
        ring->timing_pending = false;
        ring->timing_current = false;
    }
}

bool ggml_cuda_kv_stream_transfer_ring_observe_decode_latency(
        ggml_cuda_kv_stream_transfer_ring * ring, double elapsed_ms) {
    if (ring == nullptr) {
        return false;
    }
    if (ring->forced_decode_span_pages != 0) {
        return true;
    }
    const bool was_selected = ring->span_tuner.selected();
    ring->span_tuner.observe(
        elapsed_ms, ring->last_graph_decode && ring->last_graph_streamed,
        ring->last_graph_copy_batch_greedy);
    if (!was_selected && ring->span_tuner.selected()) {
        GGML_LOG_WARN(
            "%s: selected %s decode copy batches from end-to-end latency "
            "(fixed %u-page %.3f ms, greedy %u-page %.3f ms)\n",
            __func__, ring->span_tuner.use_greedy_batch() ? "greedy" : "fixed",
            KV_STREAM_COPY_BATCH_PAGES, ring->span_tuner.fixed_average_ms(),
            ring->active_slots, ring->span_tuner.greedy_average_ms());
    }
    return true;
}

ggml_cuda_kv_stream_transfer_stats ggml_cuda_kv_stream_transfer_ring_get_stats(
        const ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return {};
    }
    return {
        ring->asynchronous_page_uploads,
        ring->host_to_device_copy_commands,
        ring->compute_stream_waits,
        ring->stage_slot_reuses,
        ring->skipped_pages,
        ring->cross_layer_prefetches,
        ring->deadline_counters_host[0],
        ring->deadline_counters_host[1],
        ring->ring_peak_occupancy,
        ring->cpu_pages,
        ring->cpu_jobs,
    };
}

struct ggml_cuda_kv_stream_resident_cache {
    char * pool_data = nullptr;
    size_t pool_bytes = 0;
    size_t scratch_bytes = 0;
    size_t page_bytes = 0;
    uint32_t layer_count = 0;
    uint32_t page_tokens = 0;
    uint32_t resident_pages_per_layer = 0;
    uint32_t decode_active_pages = 0;
    uint32_t next_layer = 0;
    bool relearn_layers = true;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    std::unordered_map<const void *, uint32_t> layer_by_k;
    std::unordered_map<const void *, uint32_t> layer_by_data;
    std::unordered_map<const void *, void *> mirror_by_data;
    std::vector<uint8_t> loaded;
    std::vector<uint8_t> dirty;
    std::vector<uint8_t> precise_dirty_tracking;
    std::vector<int64_t> dirty_rows;
    std::vector<uint32_t> mutable_pages;
    bool all_pages_mutable = false;
    ggml_cuda_kv_stream_resident_stats stats;
};

static bool kv_stream_resident_cache_layout(
        const ggml_cuda_kv_stream_resident_cache * cache,
        size_t scratch_bytes,
        uint32_t decode_active_pages,
        uint32_t & resident_pages_per_layer,
        std::vector<uint32_t> & layer_pages,
        std::vector<size_t> & layer_offsets) {
    if (cache == nullptr || scratch_bytes > cache->pool_bytes ||
            scratch_bytes%cache->page_bytes != 0) {
        return false;
    }
    const size_t resident_pages_total =
        (cache->pool_bytes - scratch_bytes)/cache->page_bytes;
    const size_t pages_per_layer = resident_pages_total/cache->layer_count;
    if (pages_per_layer > UINT32_MAX ||
            pages_per_layer > std::numeric_limits<size_t>::max()/cache->layer_count) {
        return false;
    }

    resident_pages_per_layer = uint32_t(pages_per_layer);
    const size_t controlled_resident_pages = pages_per_layer*cache->layer_count;
    layer_pages.assign(cache->layer_count, resident_pages_per_layer);
    if (decode_active_pages > resident_pages_per_layer) {
        const uint64_t total_active_pages =
            uint64_t(decode_active_pages)*cache->layer_count;
        if (total_active_pages < controlled_resident_pages) {
            return false;
        }
        const uint64_t streamed_pages =
            total_active_pages - controlled_resident_pages;
        const uint64_t ring_pages = scratch_bytes/cache->page_bytes;
        if (ring_pages == 0 || streamed_pages == 0) {
            return false;
        }

        // Reduce the number of split-attention layers while each streamed
        // working set fits in both the shared ring and the active pages owned
        // by one layer. When even one split per model layer exceeds the ring,
        // that layer streams in multiple waves. Spread split layers across
        // model order so resident layers provide prefetch windows.
        const auto ceil_div = [](uint64_t numerator, uint64_t denominator) {
            return numerator/denominator + (numerator%denominator != 0);
        };
        const uint64_t splits_for_ring = ceil_div(streamed_pages, ring_pages);
        const uint64_t splits_for_layer_capacity =
            ceil_div(streamed_pages, decode_active_pages);
        const uint64_t ring_bounded_splits =
            std::min<uint64_t>(cache->layer_count, splits_for_ring);
        const uint64_t split_layers_wide =
            std::max(ring_bounded_splits, splits_for_layer_capacity);
        if (split_layers_wide == 0 || split_layers_wide > cache->layer_count) {
            return false;
        }
        const uint32_t split_layers = uint32_t(split_layers_wide);
        const uint64_t base_streamed = streamed_pages/split_layers;
        const uint64_t remainder = streamed_pages%split_layers;
        layer_pages.assign(cache->layer_count, decode_active_pages);
        for (uint32_t split = 0; split < split_layers; ++split) {
            const uint32_t layer = uint32_t(
                uint64_t(split)*cache->layer_count/split_layers);
            const uint64_t layer_streamed = base_streamed + (split < remainder ? 1 : 0);
            if (layer_streamed > decode_active_pages) {
                return false;
            }
            layer_pages[layer] = decode_active_pages - uint32_t(layer_streamed);
        }
    }

    layer_offsets.assign(size_t(cache->layer_count) + 1, 0);
    for (uint32_t layer = 0; layer < cache->layer_count; ++layer) {
        layer_offsets[layer + 1] = layer_offsets[layer] + layer_pages[layer];
    }
    return layer_offsets.back() == controlled_resident_pages;
}

static size_t kv_stream_resident_index(
        const ggml_cuda_kv_stream_resident_cache * cache,
        uint32_t layer,
        uint32_t page) {
    GGML_ASSERT(layer < cache->layer_count && page < cache->layer_pages[layer]);
    return cache->layer_offsets[layer] + page;
}

ggml_cuda_kv_stream_resident_cache * ggml_cuda_kv_stream_resident_cache_new(
        void * pool_data, size_t pool_bytes, size_t scratch_bytes, size_t page_bytes,
        uint32_t layer_count, uint32_t page_tokens) {
    if (pool_data == nullptr || scratch_bytes == 0 || scratch_bytes >= pool_bytes ||
            page_bytes == 0 || scratch_bytes%page_bytes != 0 ||
            layer_count == 0 || page_tokens != GGML_CUDA_KV_STREAM_PAGE_TOKENS) {
        return nullptr;
    }

    const size_t resident_pages = (pool_bytes - scratch_bytes)/(page_bytes*layer_count);
    if (resident_pages == 0 || resident_pages > UINT32_MAX) {
        return nullptr;
    }

    auto * cache = new ggml_cuda_kv_stream_resident_cache;
    cache->pool_data = static_cast<char *>(pool_data);
    cache->pool_bytes = pool_bytes;
    cache->scratch_bytes = scratch_bytes;
    cache->page_bytes = page_bytes;
    cache->layer_count = layer_count;
    cache->page_tokens = page_tokens;
    cache->resident_pages_per_layer = resident_pages;
    cache->layer_pages.assign(layer_count, uint32_t(resident_pages));
    cache->layer_offsets.resize(size_t(layer_count) + 1);
    for (uint32_t layer = 0; layer <= layer_count; ++layer) {
        cache->layer_offsets[layer] = size_t(layer)*resident_pages;
    }
    cache->loaded.resize(cache->layer_offsets.back(), 0);
    cache->dirty.resize(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.resize(layer_count, 0);
    return cache;
}

void ggml_cuda_kv_stream_resident_cache_free(ggml_cuda_kv_stream_resident_cache * cache) {
    delete cache;
}

void ggml_cuda_kv_stream_resident_cache_reset(ggml_cuda_kv_stream_resident_cache * cache) {
    if (cache == nullptr) {
        return;
    }
    std::fill(cache->loaded.begin(), cache->loaded.end(), 0);
    std::fill(cache->dirty.begin(), cache->dirty.end(), 0);
    std::fill(cache->precise_dirty_tracking.begin(), cache->precise_dirty_tracking.end(), 0);
    cache->dirty_rows.clear();
    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    cache->stats = {};
}

bool ggml_cuda_kv_stream_resident_cache_reconfigure(
        ggml_cuda_kv_stream_resident_cache * cache,
        size_t scratch_bytes,
        uint32_t active_pages_per_layer) {
    if (cache == nullptr) {
        return false;
    }

    uint32_t pages_per_layer = 0;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    if (!kv_stream_resident_cache_layout(
            cache, scratch_bytes, active_pages_per_layer,
            pages_per_layer, layer_pages, layer_offsets)) {
        return false;
    }

    const bool scratch_changed = cache->scratch_bytes != scratch_bytes;
    const bool layout_changed = cache->layer_pages != layer_pages;
    cache->decode_active_pages = active_pages_per_layer;
    cache->resident_pages_per_layer = pages_per_layer;
    if (!scratch_changed && !layout_changed) {
        return true;
    }

    // Publish the scratch boundary and concentrated decode layout together.
    // Both changes alter physical K/V addresses, so one invalidation is
    // required; applying them separately would discard and reload the same
    // resident working set twice.
    cache->scratch_bytes = scratch_bytes;
    cache->layer_pages = std::move(layer_pages);
    cache->layer_offsets = std::move(layer_offsets);
    cache->loaded.assign(cache->layer_offsets.back(), 0);
    cache->dirty.assign(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.assign(cache->layer_count, 0);
    cache->dirty_rows.clear();
    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    if (scratch_changed) {
        cache->stats = {};
    }
    return true;
}


bool ggml_cuda_kv_stream_resident_cache_repartition(
        ggml_cuda_kv_stream_resident_cache * cache, size_t scratch_bytes) {
    if (cache == nullptr) {
        return false;
    }
    uint32_t pages_per_layer = 0;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    if (!kv_stream_resident_cache_layout(
            cache, scratch_bytes, cache->decode_active_pages,
            pages_per_layer, layer_pages, layer_offsets)) {
        return false;
    }
    if (cache->scratch_bytes == scratch_bytes && cache->layer_pages == layer_pages) {
        return true;
    }
    cache->scratch_bytes = scratch_bytes;
    cache->resident_pages_per_layer = pages_per_layer;
    cache->layer_pages = std::move(layer_pages);
    cache->layer_offsets = std::move(layer_offsets);
    cache->loaded.assign(cache->layer_offsets.back(), 0);
    cache->dirty.assign(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.assign(cache->layer_count, 0);
    cache->dirty_rows.clear();
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    cache->stats = {};
    return true;
}

bool ggml_cuda_kv_stream_resident_cache_set_decode_layout(
        ggml_cuda_kv_stream_resident_cache * cache,
        uint32_t active_pages_per_layer) {
    if (cache == nullptr) {
        return false;
    }
    uint32_t pages_per_layer = 0;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    if (!kv_stream_resident_cache_layout(
            cache, cache->scratch_bytes, active_pages_per_layer,
            pages_per_layer, layer_pages, layer_offsets)) {
        return false;
    }
    if (cache->layer_pages == layer_pages) {
        cache->decode_active_pages = active_pages_per_layer;
        return true;
    }
    // Resident storage uses separate contiguous K and V planes per layer.
    // A layout change moves both the layer base and the V-plane boundary, so
    // migrating page-sized byte slots cannot preserve logical K/V pages.
    // Reload lazily from the authoritative host cache instead.
    cache->layer_pages = std::move(layer_pages);
    cache->layer_offsets = std::move(layer_offsets);
    cache->loaded.assign(cache->layer_offsets.back(), 0);
    cache->dirty.assign(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.assign(cache->layer_count, 0);
    cache->dirty_rows.clear();
    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    cache->decode_active_pages = active_pages_per_layer;
    cache->resident_pages_per_layer = pages_per_layer;
    return true;
}

uint32_t ggml_cuda_kv_stream_resident_cache_pages_per_layer(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? 0 : cache->resident_pages_per_layer;
}

uint32_t ggml_cuda_kv_stream_resident_cache_decode_active_pages(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? 0 : cache->decode_active_pages;
}

ggml_cuda_kv_stream_resident_stats ggml_cuda_kv_stream_resident_cache_get_stats(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? ggml_cuda_kv_stream_resident_stats{} : cache->stats;
}

bool ggml_cuda_kv_stream_resident_cache_mark_dirty_rows(
        ggml_cuda_kv_stream_resident_cache * cache,
        const int64_t * rows, size_t count) {
    if (cache == nullptr || (rows == nullptr && count != 0)) {
        return false;
    }

    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    cache->relearn_layers = true;
    if (count == 0) {
        cache->dirty_rows.clear();
    } else {
        cache->dirty_rows.assign(rows, rows + count);
    }
    std::fill(cache->dirty.begin(), cache->dirty.end(), 0);
    for (uint32_t layer = 0; layer < cache->layer_count; ++layer) {
        cache->precise_dirty_tracking[layer] = 1;
    }
    for (size_t i = 0; i < count; ++i) {
        if (rows[i] < 0) {
            cache->all_pages_mutable = true;
            std::fill(cache->dirty.begin(), cache->dirty.end(), 1);
            return true;
        }
        const uint32_t page = uint32_t(uint64_t(rows[i])/cache->page_tokens);
        const auto it = std::lower_bound(cache->mutable_pages.begin(), cache->mutable_pages.end(), page);
        if (it == cache->mutable_pages.end() || *it != page) {
            cache->mutable_pages.insert(it, page);
        }
        for (uint32_t layer = 0; layer < cache->layer_count; ++layer) {
            if (page < cache->layer_pages[layer]) {
                cache->dirty[kv_stream_resident_index(cache, layer, page)] = 1;
            }
        }
    }
    return true;
}

bool ggml_cuda_kv_stream_resident_cache_all_layers_fit(
        const ggml_cuda_kv_stream_resident_cache * cache,
        uint32_t active_pages) {
    if (cache == nullptr || active_pages == 0) {
        return false;
    }
    return std::all_of(cache->layer_pages.begin(), cache->layer_pages.end(),
        [active_pages](uint32_t pages) { return pages >= active_pages; });
}

bool ggml_cuda_kv_stream_resident_cache_get_mirror(
        ggml_cuda_kv_stream_resident_cache * cache,
        const ggml_tensor * target,
        void ** data) {
    if (cache == nullptr || target == nullptr || data == nullptr ||
            cache->dirty_rows.empty()) {
        return false;
    }
    const auto layer_it = cache->layer_by_data.find(target->data);
    const auto mirror_it = cache->mirror_by_data.find(target->data);
    if (layer_it == cache->layer_by_data.end() || mirror_it == cache->mirror_by_data.end()) {
        return false;
    }
    const uint64_t capacity = uint64_t(cache->layer_pages[layer_it->second])*cache->page_tokens;
    for (const int64_t row : cache->dirty_rows) {
        if (row < 0 || uint64_t(row) >= capacity) {
            return false;
        }
    }
    *data = mirror_it->second;
    return true;
}

void ggml_cuda_kv_stream_resident_cache_mark_mirrored(
        ggml_cuda_kv_stream_resident_cache * cache,
        const ggml_tensor * target) {
    if (cache == nullptr || target == nullptr) {
        return;
    }
    const auto layer_it = cache->layer_by_data.find(target->data);
    if (layer_it == cache->layer_by_data.end()) {
        return;
    }
    const uint32_t layer = layer_it->second;
    for (const int64_t row : cache->dirty_rows) {
        if (row < 0) {
            return;
        }
        const uint32_t page = uint32_t(uint64_t(row)/cache->page_tokens);
        if (page < cache->layer_pages[layer]) {
            cache->dirty[kv_stream_resident_index(cache, layer, page)] = 0;
        }
    }
}

void ggml_cuda_kv_stream_resident_cache_mark_dirty(
        ggml_cuda_kv_stream_resident_cache * cache,
        const ggml_tensor * target, const ggml_tensor * indices) {
    if (cache == nullptr || target == nullptr || indices == nullptr) {
        return;
    }
    const auto layer_it = cache->layer_by_data.find(target->data);
    if (layer_it == cache->layer_by_data.end()) {
        return;
    }
    const uint32_t layer = layer_it->second;
    GGML_ASSERT(layer < cache->layer_count);
    if (indices->buffer == nullptr || !ggml_backend_buffer_is_host(indices->buffer) ||
            !ggml_is_contiguous(indices) ||
            (indices->type != GGML_TYPE_I32 && indices->type != GGML_TYPE_I64)) {
        return;
    }

    cache->precise_dirty_tracking[layer] = 1;
    const size_t begin = cache->layer_offsets[layer];
    const size_t end = cache->layer_offsets[layer + 1];
    const size_t count = ggml_nelements(indices);
    for (size_t i = 0; i < count; ++i) {
        const int64_t row = indices->type == GGML_TYPE_I32 ?
            static_cast<const int32_t *>(indices->data)[i] :
            static_cast<const int64_t *>(indices->data)[i];
        if (row < 0) {
            std::fill(cache->dirty.begin() + begin, cache->dirty.begin() + end, 1);
            return;
        }
        const uint64_t page = uint64_t(row)/cache->page_tokens;
        if (page < cache->layer_pages[layer]) {
            cache->dirty[begin + page] = 1;
        }
    }
}

static uint32_t kv_stream_resident_layer(
        ggml_cuda_kv_stream_resident_cache * cache, const void * k_key) {
    GGML_ASSERT(cache != nullptr);
    auto [it, inserted] = cache->layer_by_k.emplace(k_key, cache->next_layer);
    if (inserted) {
        GGML_ASSERT(cache->next_layer < cache->layer_count);
        ++cache->next_layer;
    }
    return it->second;
}

namespace {

constexpr int KV_STREAM_MAX_PARTS_PER_CHUNK = 16;
constexpr int KV_STREAM_QUERY_WORKSPACE_TOKENS = 256;

static int kv_stream_parts_per_chunk() {
    static const int parts = []() {
        const char * value = getenv("GGML_CUDA_KV_STREAM_PARTS");
        const int parsed = value == nullptr ? 16 : atoi(value);
        return parsed == 2 || parsed == 4 || parsed == 8 || parsed == 16 ? parsed : 16;
    }();
    return parts;
}

static int64_t kv_stream_block_tokens(const ggml_tensor * dst, size_t stage_bytes) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const size_t k_row_bytes = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row_bytes = ggml_row_size(V->type, V->ne[0]);
    const size_t bytes_per_token = k_row_bytes*K->ne[2] + v_row_bytes*V->ne[2];
    if (bytes_per_token == 0) {
        return 0;
    }

    int64_t tokens = std::min<int64_t>(K->ne[1], stage_bytes/bytes_per_token);
    tokens = tokens/FATTN_KQ_STRIDE*FATTN_KQ_STRIDE;
    while (tokens > 0) {
        const size_t k_bytes = k_row_bytes*tokens*K->ne[2];
        const size_t v_offset = GGML_PAD(k_bytes, 128);
        const size_t v_bytes = v_row_bytes*tokens*V->ne[2];
        if (v_offset <= stage_bytes && v_bytes <= stage_bytes - v_offset) {
            return tokens;
        }
        tokens -= FATTN_KQ_STRIDE;
    }
    return 0;
}

template<int D>
static __global__ void kv_stream_accumulate_chunk_results(
        const float * parts,
        const float2 * meta,
        float * accumulator,
        float2 * accumulator_meta,
        int nrows,
        bool initialize,
        int nparts) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= nrows || tid >= D) {
        return;
    }
    ggml_cuda_pdl_sync();

    __shared__ float old_maximum;
    __shared__ float maximum;
    __shared__ float old_scale;
    __shared__ float denominator;

    const int base = row*nparts;
    if (tid == 0) {
        old_maximum = initialize ? -FLT_MAX : accumulator_meta[row].x;
        maximum = old_maximum;
        for (int part = 0; part < nparts; ++part) {
            maximum = fmaxf(maximum, meta[base + part].x);
        }

        old_scale = initialize ? 0.0f : expf(old_maximum - maximum);
        denominator = initialize ? 0.0f : old_scale*accumulator_meta[row].y;
        for (int part = 0; part < nparts; ++part) {
            const float weight = expf(meta[base + part].x - maximum);
            denominator += weight*meta[base + part].y;
        }
        accumulator_meta[row] = make_float2(maximum, denominator);
    }
    __syncthreads();

    float numerator = initialize ? 0.0f : old_scale*accumulator[row*D + tid];
    for (int part = 0; part < nparts; ++part) {
        const float weight = expf(meta[base + part].x - maximum);
        numerator += weight*parts[(base + part)*D + tid];
    }
    accumulator[row*D + tid] = numerator;
}

template<int D>
static __global__ void kv_stream_normalize_chunk_results(
        const float * accumulator,
        const float2 * accumulator_meta,
        float * dst,
        int nrows) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= nrows || tid >= D) {
        return;
    }
    ggml_cuda_pdl_sync();

    dst[row*D + tid] = accumulator[row*D + tid]/accumulator_meta[row].y;
}

#ifdef GGML_CUDA_FA_ALL_QUANTS
using kv_stream_native_partial_fn = void (*)(
    ggml_backend_cuda_context &, ggml_tensor *, float *, float2 *, int);

template<ggml_type type_K>
static kv_stream_native_partial_fn kv_stream_resolve_native_partial_for_v(ggml_type type_v) {
    switch (type_v) {
        case GGML_TYPE_F16:
            return &ggml_cuda_flash_attn_ext_vec_partial_case<
                GGML_CUDA_KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_F16>;
        case GGML_TYPE_Q4_0:
            return &ggml_cuda_flash_attn_ext_vec_partial_case<
                GGML_CUDA_KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_Q4_0>;
        case GGML_TYPE_Q4_1:
            return &ggml_cuda_flash_attn_ext_vec_partial_case<
                GGML_CUDA_KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_Q4_1>;
        case GGML_TYPE_Q5_0:
            return &ggml_cuda_flash_attn_ext_vec_partial_case<
                GGML_CUDA_KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_Q5_0>;
        case GGML_TYPE_Q5_1:
            return &ggml_cuda_flash_attn_ext_vec_partial_case<
                GGML_CUDA_KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_Q5_1>;
        case GGML_TYPE_Q8_0:
            return &ggml_cuda_flash_attn_ext_vec_partial_case<
                GGML_CUDA_KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_Q8_0>;
        case GGML_TYPE_BF16:
            return &ggml_cuda_flash_attn_ext_vec_partial_case<
                GGML_CUDA_KV_STREAM_HEAD_DIM, type_K, GGML_TYPE_BF16>;
        default:
            return nullptr;
    }
}

static kv_stream_native_partial_fn kv_stream_resolve_native_partial(
        ggml_type type_k, ggml_type type_v) {
#define KV_STREAM_NATIVE_K_CASE(type_K) \
        case type_K: return kv_stream_resolve_native_partial_for_v<type_K>(type_v)
    switch (type_k) {
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_F16);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_Q4_0);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_Q4_1);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_Q5_0);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_Q5_1);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_Q8_0);
        KV_STREAM_NATIVE_K_CASE(GGML_TYPE_BF16);
        default: return nullptr;
    }
#undef KV_STREAM_NATIVE_K_CASE
}
#endif // GGML_CUDA_FA_ALL_QUANTS

} // namespace

struct ggml_backend_cuda_kv_stream_type_capabilities
ggml_backend_cuda_kv_stream_get_type_capabilities(ggml_type type) {
    ggml_backend_cuda_kv_stream_type_capabilities result{};

    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            result.classified = true;
            break;
        default:
            return result;
    }

    result.storage = type != GGML_TYPE_Q8_1 && type != GGML_TYPE_Q8_K;

    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
            result.online_write = true;
            break;
        default:
            break;
    }

    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            result.decode_f16 = true;
            break;
        default:
            break;
    }

    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            result.direct_attention = true;
            break;
        default:
            break;
    }

    result.requires_initialization =
        type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S;
    result.requires_importance_matrix =
        type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
        type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M;
    result.auxiliary = type == GGML_TYPE_Q8_1 || type == GGML_TYPE_Q8_K;
    return result;
}

ggml_backend_cuda_kv_stream_attention_mode
ggml_backend_cuda_kv_stream_get_attention_mode(ggml_type type_k, ggml_type type_v) {
    const auto capabilities_k = ggml_backend_cuda_kv_stream_get_type_capabilities(type_k);
    const auto capabilities_v = ggml_backend_cuda_kv_stream_get_type_capabilities(type_v);
#ifdef GGML_CUDA_FA_ALL_QUANTS
    if (capabilities_k.direct_attention && capabilities_v.direct_attention) {
        return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS
    if (capabilities_k.storage && capabilities_v.storage &&
            capabilities_k.online_write && capabilities_v.online_write &&
            capabilities_k.decode_f16 && capabilities_v.decode_f16 &&
            !capabilities_k.requires_importance_matrix && !capabilities_v.requires_importance_matrix) {
        return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16;
    }
    return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_UNSUPPORTED;
}

bool ggml_cuda_kv_stream_page_bytes(
        ggml_type type_k, ggml_type type_v,
        uint32_t head_dim_k, uint32_t head_dim_v, uint32_t head_count,
        uint32_t page_tokens, size_t * page_bytes) {
    if (head_dim_k == 0 || head_dim_v == 0 || head_count == 0 || page_tokens == 0 || page_bytes == nullptr) {
        return false;
    }

    const auto capabilities_k = ggml_backend_cuda_kv_stream_get_type_capabilities(type_k);
    const auto capabilities_v = ggml_backend_cuda_kv_stream_get_type_capabilities(type_v);
    if (!capabilities_k.storage || !capabilities_v.storage) {
        return false;
    }

    const uint32_t block_size_k = ggml_blck_size(type_k);
    const uint32_t block_size_v = ggml_blck_size(type_v);
    if (block_size_k == 0 || block_size_v == 0 ||
            head_dim_k % block_size_k != 0 || head_dim_v % block_size_v != 0) {
        return false;
    }

    const size_t maximum = std::numeric_limits<size_t>::max();
    const size_t row_bytes_k = ggml_row_size(type_k, head_dim_k);
    const size_t row_bytes_v = ggml_row_size(type_v, head_dim_v);
    if (row_bytes_k > maximum/head_count || row_bytes_v > maximum/head_count) {
        return false;
    }
    const size_t token_bytes_k = row_bytes_k*head_count;
    const size_t token_bytes_v = row_bytes_v*head_count;
    if (token_bytes_k > maximum/page_tokens || token_bytes_v > maximum/page_tokens) {
        return false;
    }
    const size_t page_bytes_k = token_bytes_k*page_tokens;
    const size_t page_bytes_v = token_bytes_v*page_tokens;
    if (page_bytes_k > maximum - 127) {
        return false;
    }
    const size_t page_offset_v = (page_bytes_k + 127) & ~size_t(127);
    if (page_bytes_v > maximum - page_offset_v) {
        return false;
    }

    *page_bytes = page_offset_v + page_bytes_v;
    return true;
}

bool ggml_cuda_kv_stream_workspace_bytes(
        ggml_type type_k, ggml_type type_v,
        uint32_t head_dim_k, uint32_t head_dim_v, uint32_t head_count,
        uint32_t page_tokens, size_t * workspace_bytes) {
    if (workspace_bytes == nullptr) {
        return false;
    }
    const auto mode =
        ggml_backend_cuda_kv_stream_get_attention_mode(type_k, type_v);
    if (mode == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT) {
        *workspace_bytes = 0;
        return true;
    }
    if (mode != GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16) {
        return false;
    }
    return ggml_cuda_kv_stream_page_bytes(
        GGML_TYPE_F16, GGML_TYPE_F16,
        head_dim_k, head_dim_v, head_count, page_tokens, workspace_bytes);
}

bool ggml_cuda_flash_attn_ext_streamed_supported(const ggml_tensor * dst, size_t stage_bytes) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    return Q != nullptr && K != nullptr && V != nullptr &&
        Q->type == GGML_TYPE_F32 &&
        ggml_backend_cuda_kv_stream_get_attention_mode(K->type, V->type) != GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_UNSUPPORTED &&
        Q->ne[0] == GGML_CUDA_KV_STREAM_HEAD_DIM && V->ne[0] == GGML_CUDA_KV_STREAM_HEAD_DIM &&
        Q->ne[1] >= 1 && Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 &&
        K->ne[1] == V->ne[1] && K->ne[2] == V->ne[2] &&
        K->ne[1] % FATTN_KQ_STRIDE == 0 &&
        K->nb[0] == ggml_element_size(K) && V->nb[0] == ggml_element_size(V) &&
        K->nb[1] >= ggml_row_size(K->type, K->ne[0]) &&
        V->nb[1] >= ggml_row_size(V->type, V->ne[0]) &&
        K->nb[1] == ggml_row_size(K->type, K->ne[0])*K->ne[2] &&
        K->nb[2] == ggml_row_size(K->type, K->ne[0]) &&
        V->nb[1] == ggml_row_size(V->type, V->ne[0])*V->ne[2] &&
        V->nb[2] == ggml_row_size(V->type, V->ne[0]) &&
        (mask == nullptr || (mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask))) &&
        sinks == nullptr && kv_stream_block_tokens(dst, stage_bytes) > 0;
}

namespace {

static __global__ void kv_stream_record_deadline(
        const uint32_t * ready_flag,
        uint64_t * samples,
        uint64_t * misses) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long *>(samples), 1ULL);
        if (*ready_flag == 0) {
            atomicAdd(reinterpret_cast<unsigned long long *>(misses), 1ULL);
        }
    }
}

static bool kv_stream_collect_timing(ggml_cuda_kv_stream_transfer_ring * ring) {
    if (!ring->timing_pending) {
        return true;
    }
    const cudaError_t status = cudaEventQuery(ring->eval_end);
    if (status == cudaErrorNotReady) {
        return false;
    }
    CUDA_CHECK(status);

    float eval_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&eval_ms, ring->eval_start, ring->eval_end));
    double busy_ratio = 0.0;
    if (ring->copy_sample_recorded && eval_ms > 0.0f && ring->current_epoch_uploads > 0) {
        float copy_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &copy_ms, ring->copy_sample_start, ring->copy_sample_end));
        GGML_ASSERT(ring->copy_sample_uploads > 0);
        busy_ratio = std::min(1.0, double(copy_ms)*double(ring->current_epoch_uploads)/
            (double(ring->copy_sample_uploads)*double(eval_ms)));
    }
    ring->last_copy_engine_busy_ratio = busy_ratio;
    ring->last_ring_peak_occupancy = ring->current_ring_peak_occupancy;
    ring->timing_pending = false;
    return true;
}

static void kv_stream_graph_upload(
        ggml_cuda_kv_stream_transfer_ring * ring,
        kv_stream_graph_request & request,
        uint32_t slot) {
    const kv_stream_chunk_geometry & geometry = request.geometry;
    char * k_stage = ring->pool_data + size_t(slot)*geometry.k_bytes();
    char * v_stage = ring->pool_data +
        size_t(ring->active_slots)*geometry.k_bytes() + size_t(slot)*geometry.v_bytes();
    if (ring->slot_used[slot]) {
        CUDA_CHECK(cudaStreamWaitEvent(ring->copy_stream, ring->consumed[slot], 0));
        ++ring->stage_slot_reuses;
    }
    if (request.deadline_sample) {
        CUDA_CHECK(cudaMemsetAsync(
            ring->ready_flags_device + slot, 0, sizeof(uint32_t), ring->copy_stream));
    }
    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_start, ring->copy_stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        k_stage, request.k_data + geometry.token_begin*geometry.k_token_stride(),
        geometry.k_bytes(), cudaMemcpyHostToDevice, ring->copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        v_stage, request.v_data + geometry.token_begin*geometry.v_token_stride(),
        geometry.v_bytes(), cudaMemcpyHostToDevice, ring->copy_stream));
    ring->host_to_device_copy_commands += 2;
    if (request.deadline_sample) {
        CUDA_CHECK(cudaMemsetAsync(
            ring->ready_flags_device + slot, 1, sizeof(uint32_t), ring->copy_stream));
    }
    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_end, ring->copy_stream));
        ring->copy_sample_recorded = true;
        ring->copy_sample_uploads = 1;
    }
    CUDA_CHECK(cudaEventRecord(ring->ready[slot], ring->copy_stream));
    request.slot = slot;
    request.scheduled = true;
    ring->slot_used[slot] = 1;
    request.ready_slot = slot;
    ring->slot_request[slot] = size_t(&request - ring->graph_requests.data());
    ++ring->current_occupancy;
    ring->ring_peak_occupancy = std::max(
        ring->ring_peak_occupancy, ring->current_occupancy);
    ring->current_ring_peak_occupancy = std::max(
        ring->current_ring_peak_occupancy, ring->current_occupancy);
    ++ring->asynchronous_page_uploads;
    ++ring->current_epoch_uploads;
    if (ring->graph_resident_cache != nullptr) {
        ring->graph_resident_cache->stats.host_to_device_bytes +=
            geometry.k_bytes() + geometry.v_bytes();
    }
    if (ring->current_layer != KV_STREAM_NO_LAYER && request.layer > ring->current_layer) {
        ++ring->cross_layer_prefetches;
    }
}

static bool kv_stream_graph_request_follows(
        const kv_stream_graph_request & previous,
        const kv_stream_graph_request & request) {
    return request.eligible && !request.scheduled && !request.consumed &&
        request.layer == previous.layer &&
        request.k_data == previous.k_data && request.v_data == previous.v_data &&
        request.geometry.follows(previous.geometry);
}

static void kv_stream_graph_upload_batch(
        ggml_cuda_kv_stream_transfer_ring * ring,
        size_t first_request,
        uint32_t first_slot,
        uint32_t batch_pages) {
    GGML_ASSERT(batch_pages > 1);
    GGML_ASSERT(first_request + batch_pages <= ring->graph_requests.size());
    GGML_ASSERT(first_slot + batch_pages <= ring->active_slots);
    auto & first = ring->graph_requests[first_request];

    for (uint32_t page = 0; page < batch_pages; ++page) {
        auto & request = ring->graph_requests[first_request + page];
        const uint32_t slot = first_slot + page;
        GGML_ASSERT(request.eligible && !request.scheduled && !request.consumed);
        GGML_ASSERT(ring->slot_request[slot] == KV_STREAM_NO_REQUEST);
        if (ring->slot_used[slot]) {
            CUDA_CHECK(cudaStreamWaitEvent(ring->copy_stream, ring->consumed[slot], 0));
            ++ring->stage_slot_reuses;
        }
        if (request.deadline_sample) {
            CUDA_CHECK(cudaMemsetAsync(
                ring->ready_flags_device + slot, 0, sizeof(uint32_t), ring->copy_stream));
        }
    }

    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_start, ring->copy_stream));
    }
    const kv_stream_chunk_geometry & geometry = first.geometry;
    char * k_stage = ring->pool_data + size_t(first_slot)*geometry.k_bytes();
    char * v_stage = ring->pool_data +
        size_t(ring->active_slots)*geometry.k_bytes() + size_t(first_slot)*geometry.v_bytes();
    CUDA_CHECK(cudaMemcpyAsync(
        k_stage, first.k_data + geometry.token_begin*geometry.k_token_stride(),
        size_t(batch_pages)*geometry.k_bytes(), cudaMemcpyHostToDevice, ring->copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        v_stage, first.v_data + geometry.token_begin*geometry.v_token_stride(),
        size_t(batch_pages)*geometry.v_bytes(), cudaMemcpyHostToDevice, ring->copy_stream));
    ring->host_to_device_copy_commands += 2;

    for (uint32_t page = 0; page < batch_pages; ++page) {
        auto & request = ring->graph_requests[first_request + page];
        if (request.deadline_sample) {
            CUDA_CHECK(cudaMemsetAsync(
                ring->ready_flags_device + first_slot + page, 1,
                sizeof(uint32_t), ring->copy_stream));
        }
    }
    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_end, ring->copy_stream));
        ring->copy_sample_recorded = true;
        ring->copy_sample_uploads = batch_pages;
    }
    CUDA_CHECK(cudaEventRecord(ring->ready[first_slot], ring->copy_stream));

    for (uint32_t page = 0; page < batch_pages; ++page) {
        auto & request = ring->graph_requests[first_request + page];
        const uint32_t slot = first_slot + page;
        request.slot = slot;
        request.ready_slot = first_slot;
        request.scheduled = true;
        ring->slot_used[slot] = 1;
        ring->slot_request[slot] = first_request + page;
        ++ring->current_occupancy;
        ++ring->asynchronous_page_uploads;
        ++ring->current_epoch_uploads;
        if (ring->graph_resident_cache != nullptr) {
            ring->graph_resident_cache->stats.host_to_device_bytes +=
                request.geometry.k_bytes() + request.geometry.v_bytes();
        }
        if (ring->current_layer != KV_STREAM_NO_LAYER && request.layer > ring->current_layer) {
            ++ring->cross_layer_prefetches;
        }
    }
    ring->ring_peak_occupancy = std::max(
        ring->ring_peak_occupancy, ring->current_occupancy);
    ring->current_ring_peak_occupancy = std::max(
        ring->current_ring_peak_occupancy, ring->current_occupancy);
}

static uint32_t kv_stream_graph_batch_pages(
        const ggml_cuda_kv_stream_transfer_ring * ring,
        uint32_t first_slot,
        size_t * first_request) {
    if (!ring->graph_active || first_slot >= ring->active_slots ||
            ring->slot_request[first_slot] != KV_STREAM_NO_REQUEST) {
        return 0;
    }
    uint32_t reserved = 0;
    size_t index = ring->next_request;
    for (; index < ring->graph_requests.size(); ++index) {
        const auto & request = ring->graph_requests[index];
        if (request.scheduled) {
            continue;
        }
        if (request.eligible) {
            break;
        }
        ++reserved;
    }
    if (index == ring->graph_requests.size() ||
            ring->current_occupancy + reserved >= ring->active_slots) {
        return 0;
    }

    const uint32_t maximum = std::min<uint32_t>({
        ring->graph_copy_batch_pages,
        ring->active_slots - first_slot,
        ring->active_slots - ring->current_occupancy - reserved,
        uint32_t(ring->graph_requests.size() - index),
    });
    uint32_t pages = 1;
    while (pages < maximum) {
        if (ring->slot_request[first_slot + pages] != KV_STREAM_NO_REQUEST ||
                !kv_stream_graph_request_follows(
                    ring->graph_requests[index + pages - 1],
                    ring->graph_requests[index + pages])) {
            break;
        }
        ++pages;
    }
    *first_request = index;
    return pages;
}

static uint32_t kv_stream_graph_schedule_batch(
        ggml_cuda_kv_stream_transfer_ring * ring,
        uint32_t first_slot) {
    size_t first_request = KV_STREAM_NO_REQUEST;
    const uint32_t batch_pages = kv_stream_graph_batch_pages(ring, first_slot, &first_request);
    if (batch_pages == 0) {
        return 0;
    }
    // Probe every immutable copy batch at its actual compute deadline. Mutable
    // tails are produced by this graph and are excluded from prefetch quality
    // feedback so they do not force unnecessary resident-page demotions.
    ring->graph_requests[first_request].deadline_sample =
        !ring->graph_requests[first_request].mutable_tail;
    if (batch_pages == 1) {
        kv_stream_graph_upload(
            ring, ring->graph_requests[first_request], first_slot);
    } else {
        kv_stream_graph_upload_batch(
            ring, first_request, first_slot, batch_pages);
    }
    while (ring->next_request < ring->graph_requests.size() &&
            ring->graph_requests[ring->next_request].scheduled) {
        ++ring->next_request;
    }
    return batch_pages;
}

static void kv_stream_graph_fill_free_slots(ggml_cuda_kv_stream_transfer_ring * ring) {
    for (uint32_t slot = 0; slot < ring->active_slots; ++slot) {
        if (ring->slot_request[slot] != KV_STREAM_NO_REQUEST) {
            continue;
        }
        const uint32_t scheduled = kv_stream_graph_schedule_batch(ring, slot);
        if (scheduled == 0) {
            break;
        }
        slot += scheduled - 1;
    }
}

static const kv_stream_layer_plan * kv_stream_graph_layer_begin(
        ggml_cuda_kv_stream_transfer_ring * ring,
        const void * k_key,
        cudaStream_t compute_stream) {
    if (!ring->graph_active) {
        return nullptr;
    }
    const auto layer_it = ring->graph_plans.find(k_key);
    if (layer_it == ring->graph_plans.end()) {
        return nullptr;
    }
    const kv_stream_layer_plan & plan = layer_it->second;

    ring->current_layer = plan.layer;
    CUDA_CHECK(cudaEventRecord(ring->producer_ready, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(ring->copy_stream, ring->producer_ready, 0));
    for (const size_t request : plan.mutable_requests) {
        ring->graph_requests[request].eligible = true;
    }
    kv_stream_graph_fill_free_slots(ring);
    return &plan;
}

// TODO: back to taking the pool once the skeleton counters go
static void kv_stream_cpu_split_release(void * data) {
    auto * record = static_cast<kv_stream_cpu_job_record *>(data);
    record->release_ns = kv_stream_now_ns();
    record->pool->release();
}

static uint32_t kv_stream_cpu_split_dispatch(
        ggml_cuda_kv_stream_transfer_ring * ring,
        const ggml_tensor * dst,
        const std::vector<uint32_t> & pages,
        cudaStream_t compute_stream) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    kv_stream_cpu_split & split = *ring->cpu_split;
#ifdef USE_CUDA_GRAPH
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(compute_stream, &capture_status));
    GGML_ASSERT(capture_status == cudaStreamCaptureStatusNone);
#else
    GGML_UNUSED(compute_stream);
#endif // USE_CUDA_GRAPH
    GGML_ASSERT(ggml_nbytes(Q) <= split.max_rows*GGML_CUDA_KV_STREAM_HEAD_DIM*sizeof(float));

    CUDA_CHECK(cudaStreamWaitEvent(split.side_stream, ring->producer_ready, 0));
    if (!split.graph.mask_copied) {
        GGML_ASSERT(split.graph.mask_page_end <= split.context_pages && mask->ne[1] >= Q->ne[1] &&
            int64_t(split.graph.mask_page_end)*GGML_CUDA_KV_STREAM_PAGE_TOKENS <= mask->ne[0]);
        const size_t page_bytes = GGML_CUDA_KV_STREAM_PAGE_TOKENS*sizeof(uint16_t);
        const size_t offset = size_t(split.graph.mask_page_begin)*page_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(
            reinterpret_cast<char *>(split.mask) + offset, size_t(split.context_pages)*page_bytes,
            static_cast<const char *>(mask->data) + offset, mask->nb[1],
            size_t(split.graph.mask_page_end - split.graph.mask_page_begin)*page_bytes, Q->ne[1],
            cudaMemcpyDeviceToHost, split.side_stream));
        split.graph.mask_copied = true;
        split.graph.mask_rows = Q->ne[1];
    }
    GGML_ASSERT(Q->ne[1] <= split.graph.mask_rows);
    CUDA_CHECK(cudaMemcpyAsync(split.q, Q->data, ggml_nbytes(Q), cudaMemcpyDeviceToHost, split.side_stream));

    kv_stream_cpu_job job;
    ggml_cuda_kv_stream_cpu_attn_params & p = job.params;
    p.k = static_cast<const uint8_t *>(K->data);
    p.k_token_stride = K->nb[1];
    p.k_head_stride = K->nb[2];
    p.v = static_cast<const uint8_t *>(V->data);
    p.v_token_stride = V->nb[1];
    p.v_head_stride = V->nb[2];
    p.q = split.q;
    p.q_token_stride = Q->nb[1];
    p.q_head_stride = Q->nb[2];
    p.mask = split.mask;
    p.mask_token_stride = size_t(split.context_pages)*GGML_CUDA_KV_STREAM_PAGE_TOKENS*sizeof(uint16_t);
    p.page_tokens = GGML_CUDA_KV_STREAM_PAGE_TOKENS;
    p.n_head = int(Q->ne[2]);
    p.n_head_kv = int(K->ne[2]);
    p.n_tokens = int(Q->ne[1]);
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    job.pages = pages;

    // TODO: Task B skeleton body and counters
    GGML_ASSERT(split.diag.graph_jobs < split.diag.records.size());
    kv_stream_cpu_job_record & record = split.diag.records[split.diag.graph_jobs++];
    const kv_stream_cpu_knobs & knobs = split.graph.knobs;
    job.spin = knobs.spin;
    // GB/s is bytes per ns
    job.spin_ns = knobs.spin && knobs.dram_bandwidth_gbps > 0.0 ? int64_t(double(pages.size()*ring->page_bytes)/knobs.dram_bandwidth_gbps) : 0;
    job.record = &record;
    record.layer = ring->current_layer;
    record.pages = uint32_t(pages.size());
    record.spin_ns = job.spin_ns;
    record.arm_ns = kv_stream_now_ns();

    const uint32_t id = split.pool->arm(std::move(job));
    record.queued = split.pool->queued();
    ++ring->cpu_jobs;
    // after the copies, so no worker reads Q or the mask before they land
    CUDA_CHECK(cudaLaunchHostFunc(split.side_stream, kv_stream_cpu_split_release, &record));
    return id;
}

static void kv_stream_cpu_split_join(
        const ggml_cuda_kv_stream_transfer_ring * ring,
        uint32_t job,
        cudaStream_t compute_stream) {
    const kv_stream_cpu_split & split = *ring->cpu_split;
    if (split.graph.knobs.join_blocking) {
        split.pool->wait(job);
        return;
    }
    // the wait is also what orders the folded result before the accumulate
    const bool queued = kv_stream_wait_value(compute_stream, split.done_device, job);
    GGML_ASSERT(queued);
}

static bool kv_stream_page_live(
        const ggml_cuda_kv_stream_transfer_ring * ring,
        uint32_t page,
        uint32_t nchunks) {
    if (ring->live_pages.empty()) {
        return true;
    }
    // host and device must agree on n_kv and page_tokens
    GGML_ASSERT(ring->live_pages.size() == nchunks && page < nchunks);
    return ring->live_pages[page] != 0;
}

static void kv_stream_graph_release(
        ggml_cuda_kv_stream_transfer_ring * ring,
        size_t request_index,
        cudaStream_t compute_stream) {
    GGML_ASSERT(request_index < ring->graph_requests.size());
    auto & request = ring->graph_requests[request_index];
    GGML_ASSERT(request.scheduled && !request.consumed);
    const uint32_t slot = request.slot;
    GGML_ASSERT(ring->slot_request[slot] == request_index);
    CUDA_CHECK(cudaEventRecord(ring->consumed[slot], compute_stream));
    request.consumed = true;
    ring->slot_request[slot] = KV_STREAM_NO_REQUEST;
    GGML_ASSERT(ring->current_occupancy > 0);
    --ring->current_occupancy;
}

// TODO: Task B skeleton counters; graph_end logs them while the GPU runs
static void kv_stream_cpu_split_report(kv_stream_cpu_split & split, int64_t begin_wait_ns) {
    char line[512];
    if (!split.diag.layout.empty() && split.diag.layout != split.diag.printed_layout) {
        snprintf(line, sizeof(line), "cpu layout: ring %u, pages/resident/gpu/cpu per streaming layer:",
            split.diag.layout_ring);
        std::string text = line;
        for (const auto & layer : split.diag.layout) {
            text += " " + std::to_string(layer[0]) + "/" + std::to_string(layer[1]) + "/" +
                std::to_string(layer[2]) + "/" + std::to_string(layer[3]);
        }
        split.diag.report.push_back(text + "\n");
        split.diag.printed_layout = split.diag.layout;
    }
    split.diag.layout.clear();
    if (split.diag.graph_jobs == 0 && begin_wait_ns == 0) {
        return;
    }

    uint32_t pages = 0;
    uint32_t max_queued = 0;
    for (uint32_t job = 0; job < split.diag.graph_jobs; ++job) {
        const kv_stream_cpu_job_record & record = split.diag.records[job];
        const auto after_arm = [&](int64_t ns) { return double(ns - record.arm_ns)/1e3; };
        int64_t busy_ns = 0;
        uint32_t mask_zero = 0;
        for (size_t thread = 0; thread < record.end_ns.size(); ++thread) {
            busy_ns = std::max(busy_ns, record.end_ns[thread] - record.start_ns[thread]);
            mask_zero += record.mask_zero[thread];
        }
        snprintf(line, sizeof(line), "cpu job %u: layer %u, pages %u, queued %u, release %.1f, start %.1f, "
            "kernel first %.1f, kernel last %.1f, done %.1f us after arm, busy %.1f us, spin %.1f us, mask-zero pages %u\n",
            job, record.layer, record.pages, record.queued, after_arm(record.release_ns),
            after_arm(*std::min_element(record.start_ns.begin(), record.start_ns.end())),
            after_arm(*std::min_element(record.kernel_ns.begin(), record.kernel_ns.end())),
            after_arm(*std::max_element(record.kernel_ns.begin(), record.kernel_ns.end())),
            after_arm(*std::max_element(record.end_ns.begin(), record.end_ns.end())),
            busy_ns/1e3, record.spin_ns/1e3, mask_zero);
        split.diag.report.emplace_back(line);
        pages += record.pages;
        max_queued = std::max(max_queued, record.queued);
    }
    const kv_stream_cpu_knobs & knobs = split.graph.knobs;
    snprintf(line, sizeof(line), "cpu graph: jobs %u, pages %u, max queued %u, begin wait %.1f us, body %s, join %s, "
        "part %s\n", split.diag.graph_jobs, pages, max_queued, begin_wait_ns/1e3, knobs.spin ? "spin" : "kernel",
        knobs.join_blocking ? "block" : "stream", knobs.copy_part_to_device ? "copy" : "mapped");
    split.diag.report.emplace_back(line);
    split.diag.graph_jobs = 0;
}

} // namespace

void ggml_cuda_kv_stream_graph_begin(ggml_cuda_kv_stream_transfer_ring * ring) {
    GGML_ASSERT(ring != nullptr);
    const bool timing_available = kv_stream_collect_timing(ring);
    ring->graph_active = true;
    ring->graph_decode = true;
    ring->graph_decode_span_pages = ring->forced_decode_span_pages != 0 ?
        ring->forced_decode_span_pages : KV_STREAM_DECODE_SPAN_PAGES;
    ring->graph_copy_batch_greedy = ring->span_tuner.use_greedy_batch();
    ring->graph_copy_batch_pages = ring->graph_copy_batch_greedy ?
        ring->active_slots : KV_STREAM_COPY_BATCH_PAGES;
    if (ring->cpu_split != nullptr) {
        kv_stream_cpu_split & split = *ring->cpu_split;
        const int64_t wait_begin = kv_stream_now_ns();
        const bool idle = split.pool->idle();
        split.pool->wait_idle();
        kv_stream_cpu_split_report(split, idle ? 0 : kv_stream_now_ns() - wait_begin);
        split.graph = {};
        split.graph.knobs = kv_stream_cpu_read_knobs(split.supports_stream_join);
    }
    ring->graph_layer_count = 0;
    ring->current_layer = KV_STREAM_NO_LAYER;
    ring->next_request = 0;
    ring->graph_resident_cache = nullptr;
    ring->graph_requests.clear();
    ring->graph_plans.clear();
    ring->current_occupancy = 0;
    if (timing_available) {
        ring->current_ring_peak_occupancy = 0;
        ring->current_epoch_uploads = 0;
        ring->copy_sample_recorded = false;
        ring->copy_sample_uploads = 0;
    }
    ring->timing_current = timing_available;
    std::fill(ring->slot_request.begin(), ring->slot_request.end(), KV_STREAM_NO_REQUEST);
}

bool ggml_cuda_kv_stream_graph_add_attention(
        ggml_cuda_kv_stream_transfer_ring * ring,
        ggml_cuda_kv_stream_resident_cache * resident_cache,
        const ggml_tensor * dst) {
    GGML_ASSERT(ring != nullptr);
    const ggml_tensor * K = dst == nullptr ? nullptr : dst->src[1];
    const ggml_tensor * V = dst == nullptr ? nullptr : dst->src[2];
    if (resident_cache == nullptr || dst == nullptr ||
            !ggml_cuda_flash_attn_ext_streamed_supported(dst, ring->page_bytes)) {
        return false;
    }
    if (ring->graph_resident_cache != nullptr && ring->graph_resident_cache != resident_cache) {
        return false;
    }
    const bool decode_shaped = dst->src[0]->ne[1] <= GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS;
    ring->graph_decode = ring->graph_decode && decode_shaped;
    if (!decode_shaped) {
        // Graphs are rebuilt across warmup, prompt chunks, and slot reuse.
        // Relearn pointer-to-layer identity once per prefill ubatch while the
        // resident page contents are refreshed by the local multi-token path.
        if (ring->graph_resident_cache == nullptr) {
            if (resident_cache->relearn_layers) {
                resident_cache->layer_by_k.clear();
                resident_cache->layer_by_data.clear();
                resident_cache->next_layer = 0;
                resident_cache->relearn_layers = false;
            }
            ring->graph_resident_cache = resident_cache;
        }
        const uint32_t resident_layer = kv_stream_resident_layer(resident_cache, K->data);
        resident_cache->layer_by_data[K->data] = resident_layer;
        resident_cache->layer_by_data[V->data] = resident_layer;
        char * layer_base = resident_cache->pool_data + resident_cache->scratch_bytes +
            resident_cache->layer_offsets[resident_layer]*resident_cache->page_bytes;
        resident_cache->mirror_by_data[K->data] = layer_base;
        resident_cache->mirror_by_data[V->data] = layer_base +
            size_t(resident_cache->layer_pages[resident_layer])*K->nb[1]*resident_cache->page_tokens;
        return false;
    }
    ring->graph_resident_cache = resident_cache;
    // Resident placement is stable across evaluations, while deadlines must
    // follow this graph's finalized execution order.
    const uint32_t resident_layer = kv_stream_resident_layer(resident_cache, K->data);
    resident_cache->layer_by_data[K->data] = resident_layer;
    resident_cache->layer_by_data[V->data] = resident_layer;
    char * layer_base = resident_cache->pool_data + resident_cache->scratch_bytes +
        resident_cache->layer_offsets[resident_layer]*resident_cache->page_bytes;
    resident_cache->mirror_by_data[K->data] = layer_base;
    resident_cache->mirror_by_data[V->data] = layer_base +
        size_t(resident_cache->layer_pages[resident_layer])*K->nb[1]*resident_cache->page_tokens;

    const int64_t block_tokens = resident_cache->page_tokens;
    const int nchunks = int((K->ne[1] + block_tokens - 1)/block_tokens);
    const uint32_t resident_pages = resident_cache->layer_pages[resident_layer];
    if (uint32_t(nchunks) <= resident_pages) {
        return true;
    }
    GGML_ASSERT(ring->graph_plans.find(K->data) == ring->graph_plans.end());
    kv_stream_layer_plan & plan = ring->graph_plans[K->data];
    plan.layer = ring->graph_layer_count++;
    plan.request_by_page.assign(nchunks, KV_STREAM_NO_REQUEST);
    const ggml_cuda_kv_stream_page_state page_state(uint32_t(nchunks), resident_pages, ring->live_pages,
        resident_cache->mutable_pages, resident_cache->all_pages_mutable);
    // the mask crosses to the host once per graph
    const ggml_tensor * mask = dst->src[3];
    kv_stream_cpu_graph * cpu_graph = ring->cpu_split == nullptr ? nullptr : &ring->cpu_split->graph;
    uint32_t n_cpu = 0;
    if (cpu_graph != nullptr) {
        ggml_cuda_kv_stream_cpu_split_block block = ggml_cuda_kv_stream_cpu_split_get_block(dst);
        if (block == GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_OK && cpu_graph->mask != nullptr &&
                (mask->data != cpu_graph->mask->data || mask->nb[1] != cpu_graph->mask->nb[1])) {
            block = GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_MASK_CHANGED;
        }
        if (block != GGML_CUDA_KV_STREAM_CPU_SPLIT_BLOCK_OK) {
            // wide batches returned above, so the per-batch BLOCK_QUERY_TOKENS never warns here
            ring->cpu_split->warn_block(block);
        } else {
            // TODO: Task F sets the default share; the absolute page count goes with the skeleton knobs
            n_cpu = cpu_graph->knobs.pages_per_layer;
            if (n_cpu == 0 && cpu_graph->knobs.share > 0.0f) {
                uint32_t streamed = 0;
                for (uint32_t page = resident_pages; page < uint32_t(nchunks); ++page) {
                    streamed += page_state.live(page);
                }
                n_cpu = uint32_t(std::lround(cpu_graph->knobs.share*float(streamed)));
            }
        }
    }
    std::vector<uint32_t> cpu_pages = ggml_cuda_kv_stream_select_cpu_pages(page_state, n_cpu);

    for (int chunk = 0; chunk < nchunks; ++chunk) {
        const uint32_t page = uint32_t(chunk);
        if (page < resident_pages || !page_state.live(page) ||
                std::binary_search(cpu_pages.begin(), cpu_pages.end(), page)) {
            continue;
        }
        const int64_t token_begin = chunk*block_tokens;

        kv_stream_graph_request request;
        request.k_data = static_cast<const char *>(K->data);
        request.v_data = static_cast<const char *>(V->data);
        request.geometry = kv_stream_chunk_geometry(
            K, V, token_begin, std::min<int64_t>(block_tokens, K->ne[1] - token_begin));
        request.layer = plan.layer;
        request.mutable_tail = !page_state.immutable(page);
        request.eligible = !request.mutable_tail;
        GGML_ASSERT(request.geometry.page_bytes() == ring->page_bytes);

        plan.request_by_page[page] = ring->graph_requests.size();
        if (request.mutable_tail) {
            plan.mutable_requests.push_back(plan.request_by_page[page]);
        }
        ring->graph_requests.push_back(request);
    }
    // TODO: Task B skeleton layout counter
    if (ring->cpu_split != nullptr) {
        // gpu counts pages with a GPU request, not pages the GPU attended; non-live pages have neither
        const auto gpu_pages = std::count_if(plan.request_by_page.begin(), plan.request_by_page.end(),
            [](size_t request) { return request != KV_STREAM_NO_REQUEST; });
        ring->cpu_split->diag.layout.push_back({
            uint32_t(nchunks), resident_pages, uint32_t(gpu_pages), uint32_t(cpu_pages.size())});
        ring->cpu_split->diag.layout_ring = ring->active_slots;
    }
    if (!cpu_pages.empty()) {
        cpu_graph->mask = mask;
        cpu_graph->mask_page_begin = std::min(cpu_graph->mask_page_begin, cpu_pages.front());
        cpu_graph->mask_page_end = std::max(cpu_graph->mask_page_end, cpu_pages.back() + 1);
        plan.cpu_pages = std::move(cpu_pages);
    }
    return true;
}

void ggml_cuda_kv_stream_graph_finalize(
        ggml_cuda_kv_stream_transfer_ring * ring, cudaStream_t compute_stream) {
    GGML_ASSERT(ring != nullptr);
    ring->last_graph_decode = ring->graph_decode;
    ring->last_graph_copy_batch_greedy = ring->graph_copy_batch_greedy;
    ring->last_graph_streamed = !ring->graph_requests.empty();
    if (ring->graph_requests.empty()) {
        ring->timing_current = false;
        return;
    }
    if (ring->timing_current) {
        CUDA_CHECK(cudaEventRecord(ring->eval_start, compute_stream));
    }
    kv_stream_graph_fill_free_slots(ring);
}

void ggml_cuda_kv_stream_graph_end(
        ggml_cuda_kv_stream_transfer_ring * ring, cudaStream_t compute_stream) {
    GGML_ASSERT(ring != nullptr);
    if (ring->timing_current) {
        CUDA_CHECK(cudaEventRecord(ring->eval_end, compute_stream));
        ring->timing_pending = true;
        ring->timing_current = false;
    }
    // TODO: Task B skeleton counters
    if (ring->cpu_split != nullptr) {
        for (const std::string & line : ring->cpu_split->diag.report) {
            GGML_LOG_WARN("%s", line.c_str());
        }
        ring->cpu_split->diag.report.clear();
    }
}

double ggml_cuda_kv_stream_copy_engine_busy_ratio(
        ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return 0.0;
    }
    (void) kv_stream_collect_timing(ring);
    return ring->last_copy_engine_busy_ratio;
}

uint32_t ggml_cuda_kv_stream_last_ring_peak_occupancy(
        const ggml_cuda_kv_stream_transfer_ring * ring) {
    return ring == nullptr ? 0 : ring->last_ring_peak_occupancy;
}

void ggml_cuda_flash_attn_ext_streamed(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        ggml_cuda_kv_stream_transfer_ring * transfer_ring,
        ggml_cuda_kv_stream_resident_cache * resident_cache) {
    GGML_ASSERT(transfer_ring != nullptr);
    void * stage_data = transfer_ring->pool_data;
    const size_t stage_bytes = transfer_ring->page_bytes;
    GGML_ASSERT(ggml_cuda_flash_attn_ext_streamed_supported(dst, stage_bytes));

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const auto attention_mode =
        ggml_backend_cuda_kv_stream_get_attention_mode(K->type, V->type);
    const bool convert_to_f16 =
        attention_mode == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16;
#ifdef GGML_CUDA_FA_ALL_QUANTS
    const kv_stream_native_partial_fn native_partial =
        attention_mode == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT ?
            kv_stream_resolve_native_partial(K->type, V->type) : nullptr;
    GGML_ASSERT(attention_mode != GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT ||
        native_partial != nullptr);
#endif // GGML_CUDA_FA_ALL_QUANTS
    const to_fp16_cuda_t converter_k = convert_to_f16 && K->type != GGML_TYPE_F16 ?
        ggml_get_to_fp16_cuda(K->type) : nullptr;
    const to_fp16_cuda_t converter_v = convert_to_f16 && V->type != GGML_TYPE_F16 ?
        ggml_get_to_fp16_cuda(V->type) : nullptr;
    GGML_ASSERT(!convert_to_f16 || K->type == GGML_TYPE_F16 || converter_k != nullptr);
    GGML_ASSERT(!convert_to_f16 || V->type == GGML_TYPE_F16 || converter_v != nullptr);
    const int64_t block_tokens = resident_cache != nullptr ?
        resident_cache->page_tokens : kv_stream_block_tokens(dst, stage_bytes);
    const int nchunks = (K->ne[1] + block_tokens - 1)/block_tokens;
    const int nrows = ggml_nrows(dst);
    const uint32_t maximum_streamed_span_pages = Q->ne[1] <= GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS ?
        transfer_ring->graph_decode_span_pages : UINT32_MAX;

    enum class chunk_kind : uint8_t {
        resident,
        streamed,
        skipped,
        cpu,
    };

    struct chunk_descriptor {
        kv_stream_chunk_geometry geometry;
        char * k_stage = nullptr;
        char * v_stage = nullptr;
        chunk_kind kind = chunk_kind::resident;
        bool upload = true;
        bool resident_refresh = false;
        uint32_t slot = 0;
        size_t request_index = KV_STREAM_NO_REQUEST;
    };

    uint32_t resident_layer = 0;
    if (resident_cache != nullptr) {
        resident_layer = kv_stream_resident_layer(resident_cache, K->data);
    }
    const uint32_t resident_layer_pages = resident_cache == nullptr ?
        0 : resident_cache->layer_pages[resident_layer];
    const kv_stream_layer_plan * plan = kv_stream_graph_layer_begin(
        transfer_ring, K->data, ctx.stream());
    const bool graph_planned = plan != nullptr;
    GGML_ASSERT(plan == nullptr || size_t(nchunks) == plan->request_by_page.size());
    std::vector<chunk_descriptor> chunks(nchunks);
    const bool has_cpu_pages = plan != nullptr && !plan->cpu_pages.empty();
    uint32_t cpu_job = 0;
    if (has_cpu_pages) {
        GGML_ASSERT(resident_cache == transfer_ring->graph_resident_cache);
        const ggml_cuda_kv_stream_page_state page_state(uint32_t(nchunks), resident_layer_pages,
            transfer_ring->live_pages, resident_cache->mutable_pages, resident_cache->all_pages_mutable);
        for (const uint32_t page : plan->cpu_pages) {
            GGML_ASSERT(page_state.immutable(page));
            chunks[page].kind = chunk_kind::cpu;
        }
        transfer_ring->cpu_pages += plan->cpu_pages.size();
        cpu_job = kv_stream_cpu_split_dispatch(transfer_ring, dst, plan->cpu_pages, ctx.stream());
    }

    std::vector<size_t> streamed_chunks;
    streamed_chunks.reserve(nchunks);

    // live resident pages form one span; pages outside it are skipped
    uint32_t resident_first = UINT32_MAX;
    uint32_t resident_last = 0;
    for (uint32_t page = 0; page < std::min<uint32_t>(uint32_t(nchunks), resident_layer_pages); ++page) {
        if (!kv_stream_page_live(transfer_ring, page, uint32_t(nchunks))) {
            continue;
        }
        resident_first = std::min(resident_first, page);
        resident_last = page;
    }

    for (int chunk = 0; chunk < nchunks; ++chunk) {
        auto & desc = chunks[chunk];
        const int64_t token_begin = chunk*block_tokens;
        desc.geometry = kv_stream_chunk_geometry(
            K, V, token_begin, std::min<int64_t>(block_tokens, K->ne[1] - token_begin));
        const kv_stream_chunk_geometry & geometry = desc.geometry;
        GGML_ASSERT(geometry.page_bytes() <= stage_bytes);
        desc.k_stage = static_cast<char *>(stage_data);
        desc.v_stage = desc.k_stage + geometry.v_offset();

        if (resident_cache != nullptr) {
            GGML_ASSERT(geometry.token_count == resident_cache->page_tokens);
            GGML_ASSERT(geometry.page_bytes() == resident_cache->page_bytes);

            const uint32_t page = token_begin/resident_cache->page_tokens;
            if (page < resident_layer_pages) {
                // the planner hands the CPU only non-resident pages
                GGML_ASSERT(desc.kind != chunk_kind::cpu);
                if (page < resident_first || page > resident_last) {
                    desc.kind = chunk_kind::skipped;
                    ++transfer_ring->skipped_pages;
                    continue;
                }
                const size_t resident_index =
                    kv_stream_resident_index(resident_cache, resident_layer, page);
                // Keep each layer's resident K and V in separate token-major
                // planes so resident pages form one directly consumable span.
                char * layer_base = resident_cache->pool_data + resident_cache->scratch_bytes +
                    resident_cache->layer_offsets[resident_layer]*resident_cache->page_bytes;
                const size_t resident_k_plane_bytes =
                    size_t(resident_layer_pages)*geometry.k_bytes();
                desc.k_stage = layer_base + size_t(page)*geometry.k_bytes();
                desc.v_stage = layer_base + resident_k_plane_bytes + size_t(page)*geometry.v_bytes();
                if (resident_cache->loaded[resident_index]) {
                    ++resident_cache->stats.resident_hits;
                    desc.upload = resident_cache->precise_dirty_tracking[resident_layer] ?
                        resident_cache->dirty[resident_index] :
                        (dst->src[0]->ne[1] > GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS || chunk == nchunks - 1);
                    desc.resident_refresh = desc.upload;
                } else {
                    ++resident_cache->stats.resident_misses;
                    resident_cache->loaded[resident_index] = 1;
                }
            } else if (desc.kind == chunk_kind::cpu) {
                continue;
            } else if (!kv_stream_page_live(transfer_ring, page, uint32_t(nchunks))) {
                desc.kind = chunk_kind::skipped;
                ++transfer_ring->skipped_pages;
            } else {
                ++resident_cache->stats.streamed_pages;
                desc.kind = chunk_kind::streamed;
            }
        } else if (!kv_stream_page_live(transfer_ring, uint32_t(chunk), uint32_t(nchunks))) {
            desc.kind = chunk_kind::skipped;
            ++transfer_ring->skipped_pages;
        } else {
            desc.kind = chunk_kind::streamed;
        }

        if (desc.kind == chunk_kind::streamed) {
            const size_t stream_index = streamed_chunks.size();
            if (graph_planned) {
                desc.request_index = plan->request_by_page[chunk];
                GGML_ASSERT(desc.request_index != KV_STREAM_NO_REQUEST);
            } else {
                desc.slot = uint32_t(stream_index%transfer_ring->active_slots);
                desc.k_stage = transfer_ring->pool_data + size_t(desc.slot)*geometry.k_bytes();
                desc.v_stage = transfer_ring->pool_data +
                    size_t(transfer_ring->active_slots)*geometry.k_bytes() +
                    size_t(desc.slot)*geometry.v_bytes();
            }
            streamed_chunks.push_back(chunk);
        }
    }
    const bool use_mma_prefill = !convert_to_f16 &&
        Q->ne[1] > GGML_CUDA_KV_STREAM_MAX_DECODE_QUERY_TOKENS &&
        Q->ne[0] == GGML_CUDA_KV_STREAM_HEAD_DIM && V->ne[0] == GGML_CUDA_KV_STREAM_HEAD_DIM &&
        mask != nullptr && Q->ne[2] % K->ne[2] == 0 && Q->ne[2]/K->ne[2] <= 8 &&
        ggml_cuda_get_best_fattn_kernel(ctx.device, dst) == BEST_FATTN_KERNEL_MMA_F16;
    const int partial_count = use_mma_prefill ? 1 : kv_stream_parts_per_chunk();
    GGML_ASSERT(partial_count > 0 && partial_count <= KV_STREAM_MAX_PARTS_PER_CHUNK);
    GGML_ASSERT(Q->ne[1] > 0 && nrows % Q->ne[1] == 0);
    const int64_t rows_per_query = nrows/Q->ne[1];
    const int64_t workspace_queries = use_mma_prefill ? Q->ne[1] :
        std::min<int64_t>(Q->ne[1], KV_STREAM_QUERY_WORKSPACE_TOKENS);
    const size_t workspace_rows = size_t(workspace_queries*rows_per_query);
    const size_t workspace_elements = workspace_rows*dst->ne[0];

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> parts(pool);
    ggml_cuda_pool_alloc<float2> meta(pool);
    ggml_cuda_pool_alloc<float> accumulator(pool);
    ggml_cuda_pool_alloc<float2> accumulator_meta(pool);
    const bool needs_partial_reduction = convert_to_f16 || has_cpu_pages ||
        (!streamed_chunks.empty() && nchunks > 1);
    if (needs_partial_reduction) {
        parts.alloc(size_t(partial_count)*workspace_elements);
        meta.alloc(size_t(partial_count)*workspace_rows);
        accumulator.alloc(ggml_nelements(dst));
        accumulator_meta.alloc(nrows);
    }

    auto upload = [&](const chunk_descriptor & desc, cudaStream_t stream) {
        if (!desc.upload) {
            return;
        }
        const kv_stream_chunk_geometry & geometry = desc.geometry;
        const int64_t token_end = geometry.token_begin + geometry.token_count;
        size_t dirty_row_count = 0;
        if (resident_cache != nullptr && desc.resident_refresh &&
                !resident_cache->all_pages_mutable) {
            for (const int64_t row : resident_cache->dirty_rows) {
                if (row >= geometry.token_begin && row < token_end) {
                    ++dirty_row_count;
                }
            }
        }
        const size_t dirty_bytes = dirty_row_count*(geometry.k_token_stride() + geometry.v_token_stride());
        if (dirty_row_count > 0 && dirty_bytes < geometry.k_bytes() + geometry.v_bytes()) {
            for (const int64_t row : resident_cache->dirty_rows) {
                if (row < geometry.token_begin || row >= token_end) {
                    continue;
                }
                const size_t page_row = size_t(row - geometry.token_begin);
                CUDA_CHECK(cudaMemcpyAsync(
                    desc.k_stage + page_row*geometry.k_token_stride(),
                    static_cast<const char *>(K->data) + row*K->nb[1],
                    geometry.k_token_stride(), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    desc.v_stage + page_row*geometry.v_token_stride(),
                    static_cast<const char *>(V->data) + row*V->nb[1],
                    geometry.v_token_stride(), cudaMemcpyHostToDevice, stream));
            }
            transfer_ring->host_to_device_copy_commands += 2*dirty_row_count;
            resident_cache->stats.host_to_device_bytes += dirty_bytes;
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            desc.k_stage, static_cast<const char *>(K->data) + geometry.token_begin*K->nb[1],
            geometry.k_bytes(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            desc.v_stage, static_cast<const char *>(V->data) + geometry.token_begin*V->nb[1],
            geometry.v_bytes(), cudaMemcpyHostToDevice, stream));
        transfer_ring->host_to_device_copy_commands += 2;
        if (resident_cache != nullptr) {
            resident_cache->stats.host_to_device_bytes += geometry.k_bytes() + geometry.v_bytes();
        }
    };

    auto schedule_streamed = [&](size_t stream_index) {
        auto & desc = chunks[streamed_chunks[stream_index]];
        const uint32_t slot = desc.slot;
        if (transfer_ring->slot_used[slot]) {
            CUDA_CHECK(cudaStreamWaitEvent(
                transfer_ring->copy_stream, transfer_ring->consumed[slot], 0));
            ++transfer_ring->stage_slot_reuses;
        }
        upload(desc, transfer_ring->copy_stream);
        CUDA_CHECK(cudaEventRecord(transfer_ring->ready[slot], transfer_ring->copy_stream));
        transfer_ring->slot_used[slot] = 1;
        ++transfer_ring->asynchronous_page_uploads;
    };

    if (!graph_planned && !streamed_chunks.empty()) {
        // SET_ROWS and all other producers for this layer are ordered before
        // this marker on the compute stream. The copy stream may then run
        // independently while attention consumes previously prepared pages.
        CUDA_CHECK(cudaEventRecord(transfer_ring->producer_ready, ctx.stream()));
        CUDA_CHECK(cudaStreamWaitEvent(
            transfer_ring->copy_stream, transfer_ring->producer_ready, 0));
        const size_t initial = std::min<size_t>(
            transfer_ring->active_slots, streamed_chunks.size());
        for (size_t i = 0; i < initial; ++i) {
            schedule_streamed(i);
        }
    }

    bool first_chunk = true;
    size_t stream_index = 0;
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        auto & desc = chunks[chunk];
        if (desc.kind == chunk_kind::skipped || desc.kind == chunk_kind::cpu) {
            continue;
        }
        uint32_t streamed_span_pages = 0;
        if (desc.kind == chunk_kind::streamed) {
            streamed_span_pages = 1;
            uint32_t ready_slot = desc.slot;
            if (graph_planned) {
                auto & request = transfer_ring->graph_requests[desc.request_index];
                GGML_ASSERT(request.scheduled && !request.consumed);
                desc.slot = request.slot;
                ready_slot = request.ready_slot;
                desc.k_stage = transfer_ring->pool_data + size_t(desc.slot)*desc.geometry.k_bytes();
                desc.v_stage = transfer_ring->pool_data +
                    size_t(transfer_ring->active_slots)*desc.geometry.k_bytes() +
                    size_t(desc.slot)*desc.geometry.v_bytes();
                if (request.deadline_sample) {
                    kv_stream_record_deadline<<<1, 1, 0, ctx.stream()>>>(
                        transfer_ring->ready_flags_device + desc.slot,
                        transfer_ring->deadline_counters_device + 0,
                        transfer_ring->deadline_counters_device + 1);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), transfer_ring->ready[ready_slot], 0));
            ++transfer_ring->compute_stream_waits;

            // Coalesce ready pages that occupy consecutive plane slots. The
            // head stride remains the full active-ring plane width, while the
            // tensor's token extent grows across adjacent slots.
            while (!convert_to_f16 &&
                    streamed_span_pages < maximum_streamed_span_pages &&
                    chunk + int(streamed_span_pages) < nchunks) {
                auto & candidate = chunks[chunk + streamed_span_pages];
                uint32_t candidate_ready_slot = candidate.slot;
                if (candidate.kind != chunk_kind::streamed) {
                    break;
                }
                if (graph_planned) {
                    auto & request = transfer_ring->graph_requests[candidate.request_index];
                    if (!request.scheduled || request.consumed) {
                        break;
                    }
                    candidate.slot = request.slot;
                    candidate_ready_slot = request.ready_slot;
                    candidate.k_stage = transfer_ring->pool_data +
                        size_t(candidate.slot)*candidate.geometry.k_bytes();
                    candidate.v_stage = transfer_ring->pool_data +
                        size_t(transfer_ring->active_slots)*candidate.geometry.k_bytes() +
                        size_t(candidate.slot)*candidate.geometry.v_bytes();
                }
                if (candidate.slot != desc.slot + streamed_span_pages ||
                        candidate.geometry.token_begin != desc.geometry.token_begin +
                            int64_t(streamed_span_pages)*block_tokens) {
                    break;
                }
                if (graph_planned) {
                    auto & request = transfer_ring->graph_requests[candidate.request_index];
                    if (request.deadline_sample) {
                        kv_stream_record_deadline<<<1, 1, 0, ctx.stream()>>>(
                            transfer_ring->ready_flags_device + candidate.slot,
                            transfer_ring->deadline_counters_device + 0,
                            transfer_ring->deadline_counters_device + 1);
                        CUDA_CHECK(cudaGetLastError());
                    }
                }
                if (candidate_ready_slot != ready_slot) {
                    CUDA_CHECK(cudaStreamWaitEvent(
                        ctx.stream(), transfer_ring->ready[candidate_ready_slot], 0));
                    ++transfer_ring->compute_stream_waits;
                    ready_slot = candidate_ready_slot;
                }
                ++streamed_span_pages;
            }
            desc.geometry.token_count = int64_t(streamed_span_pages)*block_tokens;
            if (resident_cache != nullptr) {
                ++resident_cache->stats.streamed_attention_spans;
                resident_cache->stats.streamed_pages_attended += streamed_span_pages;
            }
        } else {
            if (convert_to_f16) {
                // Generic quantized K/V is converted one page at a time into
                // the bounded workspace. Keep resident pages separate so the
                // fallback never creates a context-sized F16 allocation.
                upload(desc, ctx.stream());
                if (desc.upload && resident_cache->precise_dirty_tracking[resident_layer]) {
                    resident_cache->dirty[kv_stream_resident_index(
                        resident_cache, resident_layer, uint32_t(chunk))] = 0;
                }
                ++resident_cache->stats.resident_attention_spans;
                ++resident_cache->stats.resident_pages_attended;
            } else {
                // Native kernels consume the live resident pages in one span.
                if (uint32_t(chunk) != resident_first) {
                    continue;
                }
                if (resident_first > 0 ||
                        resident_last + 1 < std::min<uint32_t>(uint32_t(nchunks), resident_layer_pages)) {
                    GGML_ASSERT(resident_cache->precise_dirty_tracking[resident_layer]);
                }
                const uint32_t resident_span_pages = resident_last - resident_first + 1;
                for (uint32_t page = resident_first; page <= resident_last; ++page) {
                    upload(chunks[page], ctx.stream());
                    if (chunks[page].upload &&
                            resident_cache->precise_dirty_tracking[resident_layer]) {
                        resident_cache->dirty[
                            kv_stream_resident_index(resident_cache, resident_layer, page)] = 0;
                    }
                }
                desc.geometry.token_count = int64_t(resident_span_pages)*block_tokens;
                ++resident_cache->stats.resident_attention_spans;
                resident_cache->stats.resident_pages_attended += resident_span_pages;
            }
        }

        const kv_stream_chunk_geometry & geometry = desc.geometry;
        ggml_tensor staged_k = *K;
        ggml_tensor staged_v = *V;
        staged_k.data = desc.k_stage;
        staged_k.ne[1] = geometry.token_count;
        staged_k.nb[1] = geometry.k_token_stride();
        staged_k.nb[2] = geometry.k_row_bytes;
        staged_k.nb[3] = geometry.k_bytes();
        staged_v.data = desc.v_stage;
        staged_v.ne[1] = geometry.token_count;
        staged_v.nb[1] = geometry.v_token_stride();
        staged_v.nb[2] = geometry.v_row_bytes;
        staged_v.nb[3] = geometry.v_bytes();

        ggml_tensor converted_k{};
        ggml_tensor converted_v{};
        if (convert_to_f16) {
            GGML_ASSERT(transfer_ring->conversion_data != nullptr);
            const size_t k_elements = size_t(K->ne[0])*geometry.token_count*K->ne[2];
            const size_t v_elements = size_t(V->ne[0])*geometry.token_count*V->ne[2];
            const size_t k_f16_bytes = k_elements*sizeof(half);
            const size_t v_f16_offset = GGML_PAD(k_f16_bytes, 128);
            const size_t v_f16_bytes = v_elements*sizeof(half);
            GGML_ASSERT(v_f16_offset <= transfer_ring->conversion_bytes &&
                v_f16_bytes <= transfer_ring->conversion_bytes - v_f16_offset);

            auto convert_page = [&](const ggml_tensor & src, half * converted,
                    size_t elements, to_fp16_cuda_t converter) {
                if (converter == nullptr) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        converted, src.data, elements*sizeof(half),
                        cudaMemcpyDeviceToDevice, ctx.stream()));
                    return;
                }
                converter(src.data, converted, elements, ctx.stream());
            };

            auto * converted_k_data = reinterpret_cast<half *>(transfer_ring->conversion_data);
            auto * converted_v_data = reinterpret_cast<half *>(
                transfer_ring->conversion_data + v_f16_offset);
            convert_page(staged_k, converted_k_data, k_elements, converter_k);
            convert_page(staged_v, converted_v_data, v_elements, converter_v);

            converted_k = staged_k;
            converted_k.type = GGML_TYPE_F16;
            converted_k.data = converted_k_data;
            converted_k.nb[0] = sizeof(half);
            converted_k.nb[1] = size_t(K->ne[0])*K->ne[2]*sizeof(half);
            converted_k.nb[2] = size_t(K->ne[0])*sizeof(half);
            converted_k.nb[3] = converted_k.nb[1]*converted_k.ne[1];
            converted_v = staged_v;
            converted_v.type = GGML_TYPE_F16;
            converted_v.data = converted_v_data;
            converted_v.nb[0] = sizeof(half);
            converted_v.nb[1] = size_t(V->ne[0])*V->ne[2]*sizeof(half);
            converted_v.nb[2] = size_t(V->ne[0])*sizeof(half);
            converted_v.nb[3] = converted_v.nb[1]*converted_v.ne[1];
        }
        ggml_tensor staged_mask{};
        ggml_tensor * staged_mask_ptr = nullptr;
        if (mask != nullptr) {
            staged_mask = *mask;
            staged_mask.data = static_cast<char *>(mask->data) + geometry.token_begin*mask->nb[0];
            staged_mask.ne[0] = geometry.token_count;
            staged_mask_ptr = &staged_mask;
        }

        ggml_tensor staged_dst = *dst;
        staged_dst.src[1] = convert_to_f16 ? &converted_k : &staged_k;
        staged_dst.src[2] = convert_to_f16 ? &converted_v : &staged_v;
        staged_dst.src[3] = staged_mask_ptr;

        // Preserve normal CUDA flash attention when the active cache is fully resident or fits in one streamed page.
        // This avoids a partial reduction and keeps logits identical to a non-streamed cache.
        if (!convert_to_f16 && !has_cpu_pages && (streamed_chunks.empty() || nchunks == 1)) {
            ggml_cuda_flash_attn_ext(ctx, &staged_dst);
            if (desc.kind == chunk_kind::streamed) {
                if (graph_planned) {
                    kv_stream_graph_release(transfer_ring, desc.request_index, ctx.stream());
                    kv_stream_graph_fill_free_slots(transfer_ring);
                } else {
                    CUDA_CHECK(cudaEventRecord(transfer_ring->consumed[desc.slot], ctx.stream()));
                }
            }
            return;
        }

        GGML_ASSERT(needs_partial_reduction);

        // Keep each staged KV span alive while all bounded query tiles consume it.
        // This bounds vector scratch without issuing another H2D transfer for the span.
        for (int64_t query_begin = 0; query_begin < Q->ne[1]; query_begin += workspace_queries) {
            const int64_t query_count = std::min<int64_t>(
                workspace_queries, Q->ne[1] - query_begin);
            ggml_tensor query_q = *Q;
            query_q.data = static_cast<char *>(Q->data) + query_begin*Q->nb[1];
            query_q.ne[1] = query_count;

            ggml_tensor query_mask{};
            ggml_tensor * query_mask_ptr = nullptr;
            if (staged_mask_ptr != nullptr) {
                query_mask = *staged_mask_ptr;
                query_mask.data = static_cast<char *>(staged_mask_ptr->data) +
                    query_begin*mask->nb[1];
                query_mask.ne[1] = query_count;
                query_mask_ptr = &query_mask;
            }

            ggml_tensor query_dst = staged_dst;
            query_dst.ne[1] = query_count;
            query_dst.src[0] = &query_q;
            query_dst.src[3] = query_mask_ptr;

            if (use_mma_prefill) {
                ggml_cuda_flash_attn_ext_mma_f16_partial_case<
                    GGML_CUDA_KV_STREAM_HEAD_DIM, GGML_CUDA_KV_STREAM_HEAD_DIM, 8, 8>(
                        ctx, &query_dst, parts.ptr, meta.ptr);
                if (resident_cache != nullptr) {
                    ++resident_cache->stats.mma_prefill_attention_spans;
                }
            } else {
                if (convert_to_f16) {
                    ggml_cuda_flash_attn_ext_vec_partial_case<
                        GGML_CUDA_KV_STREAM_HEAD_DIM, GGML_TYPE_F16, GGML_TYPE_F16>(
                            ctx, &query_dst, parts.ptr, meta.ptr, partial_count);
                } else {
#ifdef GGML_CUDA_FA_ALL_QUANTS
                    GGML_ASSERT(native_partial != nullptr);
                    native_partial(ctx, &query_dst, parts.ptr, meta.ptr, partial_count);
#else
                    GGML_ABORT("native quantized KV streaming requires GGML_CUDA_FA_ALL_QUANTS");
#endif // GGML_CUDA_FA_ALL_QUANTS
                }
            }

            const int tile_nrows = int(query_count*rows_per_query);
            const size_t row_offset = size_t(query_begin*rows_per_query);
            const dim3 blocks(tile_nrows, 1, 1);
            const dim3 threads(GGML_CUDA_KV_STREAM_HEAD_DIM, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
            ggml_cuda_kernel_launch(
                kv_stream_accumulate_chunk_results<GGML_CUDA_KV_STREAM_HEAD_DIM>, launch_params,
                parts.ptr, meta.ptr, accumulator.ptr + row_offset*dst->ne[0],
                accumulator_meta.ptr + row_offset, tile_nrows, first_chunk, partial_count);
            CUDA_CHECK(cudaGetLastError());
        }

        first_chunk = false;

        if (desc.kind == chunk_kind::streamed) {
            for (uint32_t page = 0; page < streamed_span_pages; ++page) {
                auto & member = chunks[chunk + page];
                if (graph_planned) {
                    kv_stream_graph_release(
                        transfer_ring, member.request_index, ctx.stream());
                } else {
                    CUDA_CHECK(cudaEventRecord(
                        transfer_ring->consumed[member.slot], ctx.stream()));
                    const size_t next = stream_index + page + transfer_ring->active_slots;
                    if (next < streamed_chunks.size()) {
                        schedule_streamed(next);
                    }
                }
            }
            stream_index += streamed_span_pages;
            if (graph_planned) {
                kv_stream_graph_fill_free_slots(transfer_ring);
            }
            chunk += int(streamed_span_pages) - 1;
        }
    }
    const dim3 blocks(nrows, 1, 1);
    const dim3 threads(GGML_CUDA_KV_STREAM_HEAD_DIM, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    if (has_cpu_pages) {
        const kv_stream_cpu_split & split = *transfer_ring->cpu_split;
        GGML_ASSERT(size_t(nrows) <= split.max_rows);
        kv_stream_cpu_split_join(transfer_ring, cpu_job, ctx.stream());
        const float * cpu_result = split.result_device;
        const float2 * cpu_result_meta = split.result_meta_device;
        // TODO: Task D keeps one of the two part paths
        if (split.graph.knobs.copy_part_to_device) {
            GGML_ASSERT(size_t(nrows) <= workspace_rows);
            CUDA_CHECK(cudaMemcpyAsync(parts.ptr, split.result, size_t(nrows)*GGML_CUDA_KV_STREAM_HEAD_DIM*sizeof(float),
                cudaMemcpyHostToDevice, ctx.stream()));
            CUDA_CHECK(cudaMemcpyAsync(meta.ptr, split.result_meta, size_t(nrows)*sizeof(float2),
                cudaMemcpyHostToDevice, ctx.stream()));
            cpu_result = parts.ptr;
            cpu_result_meta = meta.ptr;
        }
        ggml_cuda_kernel_launch(kv_stream_accumulate_chunk_results<GGML_CUDA_KV_STREAM_HEAD_DIM>, launch_params,
            cpu_result, cpu_result_meta, accumulator.ptr, accumulator_meta.ptr, nrows, first_chunk, 1);
        CUDA_CHECK(cudaGetLastError());
        first_chunk = false;
    }
    GGML_ASSERT(!first_chunk);

    ggml_cuda_kernel_launch(kv_stream_normalize_chunk_results<GGML_CUDA_KV_STREAM_HEAD_DIM>, launch_params,
        accumulator.ptr, accumulator_meta.ptr, static_cast<float *>(dst->data), nrows);
    CUDA_CHECK(cudaGetLastError());
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ctx, dst)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
