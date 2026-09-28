#include "llama-kv-stream-config.h"
#include "testing.h"

#include <limits>
#include <string>

int main() {
    testing t;

    t.test("streaming is opt-in", [](testing & t) {
        llama_kv_stream_config config;
        const auto result = llama_kv_stream_config_validate(config);
        t.assert_true("disabled config is valid", result.valid);
        t.assert_true("disabled config remains disabled", !result.enabled);
        t.assert_true("disabled config does not warn", result.warning.empty());
    });

    t.test("cpu attention threads without streaming warn and stay off", [](testing & t) {
        llama_kv_stream_config config;
        config.cpu_threads = 4;
        const auto result = llama_kv_stream_config_validate(config);
        t.assert_true("config is valid", result.valid);
        t.assert_true("config remains disabled", !result.enabled);
        t.assert_true("config warns", !result.warning.empty());
    });

    t.test("supported target configuration is accepted", [](testing & t) {
        llama_kv_stream_config config;
        config.stage_bytes        = 64ULL*1024ULL*1024ULL;
        config.minimum_stage_bytes = 1664ULL*256ULL;
        config.arch_qwen35        = true;
        config.context_default    = true;
        config.single_sequence    = true;
        config.flash_attention    = true;
        config.kv_offload         = true;
        config.cpu_threads        = 4;

        const auto result = llama_kv_stream_config_validate(config);
        t.assert_true("config is valid", result.valid);
        t.assert_true("config is enabled", result.enabled);
        t.assert_true("config does not warn", result.warning.empty());
    });

    t.test("multi-sequence with unified KV is accepted and enabled", [](testing & t) {
        llama_kv_stream_config config;
        config.stage_bytes         = 64ULL*1024ULL*1024ULL;
        config.minimum_stage_bytes = 1664ULL*256ULL;
        config.arch_qwen35         = true;
        config.context_default     = true;
        config.single_sequence     = false;
        config.kv_unified          = true;
        config.flash_attention     = true;
        config.kv_offload          = true;

        const auto result = llama_kv_stream_config_validate(config);
        t.assert_true("config is valid", result.valid);
        t.assert_true("config is enabled", result.enabled);
    });

    t.test("each unsupported condition fails loudly", [](testing & t) {
        llama_kv_stream_config base;
        base.stage_bytes         = 64ULL*1024ULL*1024ULL;
        base.minimum_stage_bytes = 1664ULL*256ULL;
        base.arch_qwen35         = true;
        base.context_default     = true;
        base.single_sequence     = true;
        base.flash_attention     = true;
        base.kv_offload          = true;

        auto expect_invalid = [&](const char * name, const llama_kv_stream_config & config) {
            const auto result = llama_kv_stream_config_validate(config);
            t.assert_true(name, !result.valid && !result.enabled && !result.error.empty());
        };

        auto config = base;
        config.arch_qwen35 = false;
        expect_invalid("non-Qwen architecture", config);
        config = base;
        config.context_default = false;
        expect_invalid("draft/MTP context", config);
        config = base;
        config.single_sequence = false;
        config.kv_unified = false;
        expect_invalid("parallel sequences without unified KV", config);
        config = base;
        config.flash_attention = false;
        expect_invalid("Flash Attention disabled", config);
        config = base;
        config.kv_offload = false;
        expect_invalid("KV offload disabled", config);
        config = base;
        config.stage_bytes = config.minimum_stage_bytes - 1;
        expect_invalid("stage smaller than one page", config);
    });

    t.test("cpu share is auto or a number in [0, 1]", [](testing & t) {
        llama_kv_stream_config base;
        base.stage_bytes         = 64ULL*1024ULL*1024ULL;
        base.minimum_stage_bytes = 1664ULL*256ULL;
        base.arch_qwen35         = true;
        base.context_default     = true;
        base.single_sequence     = true;
        base.flash_attention     = true;
        base.kv_offload          = true;
        base.cpu_threads         = 4;
        t.assert_equal(-1.0f, base.cpu_share);

        auto config = base;
        config.cpu_share = 1.5f;
        auto result = llama_kv_stream_config_validate(config);
        t.assert_true("1.5 is invalid", !result.valid && !result.enabled);
        t.assert_true("the error names the share", result.error.find("share") != std::string::npos);

        config.cpu_share = std::numeric_limits<float>::quiet_NaN();
        result = llama_kv_stream_config_validate(config);
        t.assert_true("NaN is invalid", !result.valid && !result.enabled && !result.error.empty());

        for (const float share : { -1.0f, 0.4f }) {
            config.cpu_share = share;
            result = llama_kv_stream_config_validate(config);
            t.assert_true("auto and 0.4 are valid", result.valid && result.enabled && result.warning.empty());
        }
    });

    t.test("a cpu share without cpu attention threads warns", [](testing & t) {
        llama_kv_stream_config config;
        config.stage_bytes         = 64ULL*1024ULL*1024ULL;
        config.minimum_stage_bytes = 1664ULL*256ULL;
        config.arch_qwen35         = true;
        config.context_default     = true;
        config.single_sequence     = true;
        config.flash_attention     = true;
        config.kv_offload          = true;

        config.cpu_share = 0.4f;
        auto result = llama_kv_stream_config_validate(config);
        t.assert_true("0.4 with no threads is valid", result.valid && result.enabled);
        t.assert_true("0.4 with no threads warns", result.warning.find("share") != std::string::npos);

        config.cpu_share = -1.0f;
        result = llama_kv_stream_config_validate(config);
        t.assert_true("auto with no threads is valid", result.valid && result.enabled);
        t.assert_true("auto with no threads does not warn", result.warning.empty());
    });

    t.test("cpu attention threads clamp to the machine and keep zero off", [](testing & t) {
        t.assert_equal(uint32_t(0), llama_kv_stream_cpu_threads_resolve(0, 8));
        t.assert_equal(uint32_t(6), llama_kv_stream_cpu_threads_resolve(6, 8));
        t.assert_equal(uint32_t(8), llama_kv_stream_cpu_threads_resolve(64, 8));
        t.assert_equal(uint32_t(4), llama_kv_stream_cpu_threads_resolve(4, 0));
    });

    t.test("pool is partitioned evenly across layers with one scratch page", [](testing & t) {
        const auto layout = llama_kv_stream_pool_layout_make({
            /*.pool_bytes   =*/ 64ULL*1024ULL*1024ULL,
            /*.page_bytes   =*/ 1664ULL*256ULL,
            /*.layer_count  =*/ 16,
            /*.scratch_pages=*/ 1,
        });

        t.assert_true("layout is valid", layout.valid);
        t.assert_equal(uint32_t(9), layout.resident_pages_per_layer);
        t.assert_equal(uint32_t(9*256), layout.resident_tokens_per_layer);
        t.assert_equal(1664ULL*256ULL, layout.scratch_bytes);
        t.assert_equal(
            64ULL*1024ULL*1024ULL,
            layout.scratch_bytes + layout.resident_bytes + layout.unused_bytes);
    });

    t.test("pool rejects missing scratch or resident capacity", [](testing & t) {
        auto layout = llama_kv_stream_pool_layout_make({ 0, 1664ULL*256ULL, 16, 1 });
        t.assert_true("zero pool", !layout.valid);

        layout = llama_kv_stream_pool_layout_make({ 1664ULL*256ULL, 1664ULL*256ULL, 16, 1 });
        t.assert_true("scratch-only pool", !layout.valid);

        layout = llama_kv_stream_pool_layout_make({ 64ULL*1024ULL*1024ULL, 0, 16, 1 });
        t.assert_true("zero page", !layout.valid);
    });

    return t.summary();
}
