// Test for state restore with fragmented KV cache
// This tests the fix for: https://github.com/ggml-org/llama.cpp/issues/17527
// The issue was that state restore required contiguous KV cache slots,
// which fails when the cache is fragmented.
//
// The fix lets state_read_meta() fall back to find_slot(ubatch, false),
// allowing non-contiguous slot allocation. A contiguous block is still
// preferred when one is free, so a restore does not scatter needlessly.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// run count of the last state_read_data restore
static int g_restore_runs = -1;

// cell layout from LLAMA_KV_CACHE_DEBUG=3 dump
static std::string g_cell_map;

static void log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    unsigned cells = 0;
    size_t   runs  = 0;

    if (sscanf(text, "state_read_data: restoring %u cells in %zu runs", &cells, &runs) == 2) {
        g_restore_runs = (int) runs;
    }

    // parse cell layout dump
    const char * sp = strchr(text, ' ');
    if (text[0] == '\n' && text[1] != '\0' && strchr(".0123456789M", text[1]) && (sp == nullptr || sp[1] == '*')) {
        g_cell_map.clear();
        for (const char * c = text + 1; *c; ++c) {
            if (*c == '.' || *c == 'M' || (*c >= '0' && *c <= '9')) {
                g_cell_map.push_back(*c);
            }
        }
    }

    if (level != GGML_LOG_LEVEL_DEBUG) {
        fputs(text, stderr);
    }
}

static int page_seq_count(size_t page, size_t page_size) {
    std::string seen;
    for (size_t i = page*page_size; i < std::min(g_cell_map.size(), (page + 1)*page_size); ++i) {
        const char c = g_cell_map[i];
        if (c != '.' && seen.find(c) == std::string::npos) {
            seen.push_back(c);
        }
    }
    return (int) seen.size();
}

static int page_cell_count(size_t page, size_t page_size, char c) {
    int n = 0;
    for (size_t i = page*page_size; i < std::min(g_cell_map.size(), (page + 1)*page_size); ++i) {
        n += g_cell_map[i] == c;
    }
    return n;
}

static int check_pages_pure(const char * who, size_t n_pages, size_t page_size) {
    for (size_t p = 0; p < n_pages; ++p) {
        const int n = page_seq_count(p, page_size);
        if (n > 1) {
            fprintf(stderr, "%s : FAILED - page %zu holds %d sequences: %s\n", who, p, n,
                    g_cell_map.substr(p*page_size, page_size).c_str());
            return 1;
        }
    }
    return 0;
}

static int test_prefill_takes_empty_pages(llama_context * ctx, llama_batch & batch) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    for (int s = 0; s < 2; s++) {
        common_batch_clear(batch);
        for (int i = 0; i < 32; i++) {
            common_batch_add(batch, 1, i, {s}, false);
        }
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to prefill seq %d\n", __func__, s);
            return 1;
        }
    }

    llama_memory_seq_rm(mem, 0, -1, -1);

    common_batch_clear(batch);
    for (int i = 0; i < 300; i++) {
        common_batch_add(batch, 1, i, {2}, false);
    }
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to prefill seq 2\n", __func__);
        return 1;
    }

    // trigger dump to inspect prefill layout
    common_batch_clear(batch);
    common_batch_add(batch, 1, 32, {1}, false);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 1\n", __func__);
        return 1;
    }

    if (check_pages_pure(__func__, 4, 256)) {
        return 1;
    }

    fprintf(stderr, "%s : SUCCESS - prefill took empty pages only\n", __func__);
    return 0;
}

static int test_full_cache_uses_holes(llama_context * ctx, llama_batch & batch) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    const int n_prefill[3] = { 256, 256, 512 };
    for (int s = 0; s < 3; s++) {
        common_batch_clear(batch);
        for (int i = 0; i < n_prefill[s]; i++) {
            common_batch_add(batch, 1, i, {s}, false);
        }
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to prefill seq %d\n", __func__, s);
            return 1;
        }
    }

    llama_memory_seq_rm(mem, 2, 0, 100);

    for (int i = 0; i < 2; i++) {
        common_batch_clear(batch);
        common_batch_add(batch, 1, 256 + i, {0}, false);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : FAILED - decode of seq 0 failed with free cells in the cache\n", __func__);
            return 1;
        }
    }

    if (page_seq_count(2, 256) != 2) {
        fprintf(stderr, "%s : FAILED - seq 0 did not land in page 2: %s\n", __func__, g_cell_map.substr(512, 256).c_str());
        return 1;
    }

    fprintf(stderr, "%s : SUCCESS - seq 0 fell back to a hole\n", __func__);
    return 0;
}

static int test_restore_takes_empty_pages(llama_context * ctx, llama_batch & batch) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    const int n_prefill[3] = { 40, 300, 100 };
    for (int s = 0; s < 3; s++) {
        common_batch_clear(batch);
        for (int i = 0; i < n_prefill[s]; i++) {
            common_batch_add(batch, 1, i, {s}, false);
        }
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to prefill seq %d\n", __func__, s);
            return 1;
        }
    }

    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 1));
    if (llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 1) != seq_state.size()) {
        fprintf(stderr, "%s : failed to save seq 1 state\n", __func__);
        return 1;
    }

    llama_memory_seq_rm(mem, 1, -1, -1);

    common_batch_clear(batch);
    common_batch_add(batch, 1, 40, {0}, false);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 0\n", __func__);
        return 1;
    }

    if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 1) != seq_state.size()) {
        fprintf(stderr, "%s : failed to restore seq 1 state\n", __func__);
        return 1;
    }

    common_batch_clear(batch);
    common_batch_add(batch, 1, 41, {0}, false);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 0\n", __func__);
        return 1;
    }

    if (check_pages_pure(__func__, 4, 256)) {
        return 1;
    }

    fprintf(stderr, "%s : SUCCESS - restore took empty pages only\n", __func__);
    return 0;
}

static int test_concurrent_decode_pages(llama_context * ctx, llama_batch & batch) {
    llama_memory_clear(llama_get_memory(ctx), true);

    for (int i = 0; i < 32; i++) {
        common_batch_clear(batch);
        common_batch_add(batch, 1, i, {0}, false);
        common_batch_add(batch, 1, i, {1}, false);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to decode step %d\n", __func__, i);
            return 1;
        }
    }

    if (g_cell_map.size() != 1024) {
        fprintf(stderr, "%s : FAILED - no cell map captured (%zu cells)\n", __func__, g_cell_map.size());
        return 1;
    }

    if (check_pages_pure(__func__, 4, 256)) {
        return 1;
    }

    fprintf(stderr, "%s : SUCCESS - each sequence stayed in its own page\n", __func__);
    return 0;
}

static int test_decode_after_park_stays_in_page(llama_context * ctx, llama_batch & batch) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    // seq 0 fills pages 0-1, seq 1 fills page 2 exactly
    const int n_prefill[2] = { 512, 256 };
    for (int s = 0; s < 2; s++) {
        common_batch_clear(batch);
        for (int i = 0; i < n_prefill[s]; i++) {
            common_batch_add(batch, 1, i, {s}, false);
        }
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to prefill seq %d\n", __func__, s);
            return 1;
        }
    }

    llama_memory_seq_rm(mem, 0, -1, -1);

    // 8 decode steps, the 9th only triggers the dump
    for (int i = 0; i < 9; i++) {
        common_batch_clear(batch);
        common_batch_add(batch, 1, 256 + i, {1}, false);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to decode step %d\n", __func__, i);
            return 1;
        }
    }

    if (page_cell_count(0, 256, '1') != 8 || page_seq_count(1, 256) != 0) {
        fprintf(stderr, "%s : FAILED - decode tokens scattered across pages: %s\n", __func__, g_cell_map.substr(0, 512).c_str());
        return 1;
    }

    fprintf(stderr, "%s : SUCCESS - decode after park filled one page\n", __func__);
    return 0;
}

static int test_full_tail_reuses_owned_page(llama_context * ctx, llama_batch & batch) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    const int n_prefill[2] = { 256, 512 };
    for (int s = 1; s >= 0; s--) {
        common_batch_clear(batch);
        for (int i = 0; i < n_prefill[s]; i++) {
            common_batch_add(batch, 1, i, {s}, false);
        }
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to prefill seq %d\n", __func__, s);
            return 1;
        }
    }

    llama_memory_seq_rm(mem, 1, 100, 200);

    // 4 decode steps, the 5th triggers the dump
    for (int i = 0; i < 5; i++) {
        common_batch_clear(batch);
        common_batch_add(batch, 1, 512 + i, {1}, false);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to decode step %d\n", __func__, i);
            return 1;
        }
    }

    if (page_cell_count(0, 256, '1') != 160 || page_seq_count(3, 256) != 0) {
        fprintf(stderr, "%s : FAILED - decode opened a new page instead of the owned hole: page 0 holds %d, page 3 holds %d seqs\n",
                __func__, page_cell_count(0, 256, '1'), page_seq_count(3, 256));
        return 1;
    }

    fprintf(stderr, "%s : SUCCESS - full tail page fell back to the sequence's own page\n", __func__);
    return 0;
}

static int test_contiguous_first(llama_context * ctx, llama_batch & batch) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    common_batch_clear(batch);
    for (int i = 0; i < 32; i++) {
        common_batch_add(batch, 1, i, {0}, false);
        common_batch_add(batch, 1, i, {1}, false);
    }
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 0, 1\n", __func__);
        return 1;
    }

    common_batch_clear(batch);
    for (int i = 0; i < 64; i++) {
        common_batch_add(batch, 1, i, {2}, false);
    }
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 2\n", __func__);
        return 1;
    }

    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 2));
    if (llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 2) != seq_state.size()) {
        fprintf(stderr, "%s : failed to save seq 2 state\n", __func__);
        return 1;
    }

    llama_memory_seq_rm(mem, 0, -1, -1);
    llama_memory_seq_rm(mem, 2, -1, -1);

    g_restore_runs = -1;
    if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 2) != seq_state.size()) {
        fprintf(stderr, "%s : failed to restore seq 2 state\n", __func__);
        return 1;
    }

    if (g_restore_runs != 1) {
        fprintf(stderr, "%s : FAILED - seq 2 restored in %d runs, expected 1 (contiguous block was available)\n", __func__, g_restore_runs);
        return 1;
    }

    fprintf(stderr, "%s : SUCCESS - seq 2 restored into the contiguous block\n", __func__);
    return 0;
}

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 3;
    params.n_ctx = 256;

    common_init();

    llama_log_set(log_callback, nullptr);

    setenv("LLAMA_KV_CACHE_DEBUG", "3", 1);

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // init

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params, true);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init_from_model(model, common_context_params_to_llama(params));

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    GGML_UNUSED(model);

    // tokenize prompt
    std::vector<llama_token> tokens(70, 1);

    // interleave the 3 sequences:
    // 01201230123...
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        for (int s = 0; s < params.n_parallel; ++s) {
            common_batch_add(batch, tokens[i], i, {s}, false);
        }
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 0\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : processed prompt on seq 0, 1, 2 (%zu tokens each)\n", __func__, tokens.size());

    // Save state of seq 1
    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 1));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (ncopy != seq_state.size()) {
        fprintf(stderr, "%s : failed to save seq 1 state\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s : saved seq 1 state, %zu bytes\n", __func__, ncopy);

    // clear seq 1 to create a "hole" in the KV cache (fragmentation)
    // 0.20.20.20.2....
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 1, -1, -1);
    fprintf(stderr, "%s : cleared seq 1 to create fragmentation\n", __func__);

    // Now the cache has holes where seq 1 was
    // This creates fragmentation - there's no contiguous block large enough
    // for the seq 1 state if we only look for contiguous slots

    // Restore seq 1 state into seq 1 (should work with non-contiguous allocation)
    // We use seq 1 since it's a valid sequence ID (0 to n_parallel-1)
    // Before the fix, this would fail with "failed to find available cells in kv cache"
    const size_t nset = llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (nset != seq_state.size()) {
        fprintf(stderr, "%s : FAILED to restore seq state into fragmented cache (got %zu, expected %zu)\n",
                __func__, nset, seq_state.size());
        fprintf(stderr, "%s : This is the bug - state restore fails with fragmented KV cache\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    fprintf(stderr, "%s : restored state into seq 1, %zu bytes\n", __func__, nset);

    // Verify we can decode with the restored state
    // Generate one token to verify the restored state is usable
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto next_token = llama_sampler_sample(smpl, ctx, -1);
    auto next_token_str = common_token_to_piece(ctx, next_token);

    common_batch_clear(batch);
    common_batch_add(batch, next_token, (int)tokens.size(), {1}, true);

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode with restored state\n", __func__);
        llama_sampler_free(smpl);
        llama_batch_free(batch);
        return 1;
    }

    fprintf(stderr, "%s : successfully decoded with restored state, generated: '%s'\n", __func__, next_token_str.c_str());
    fprintf(stderr, "%s : SUCCESS - state restore works with fragmented KV cache\n", __func__);

    int ret = test_contiguous_first(ctx, batch);

    llama_sampler_free(smpl);
    llama_free(ctx);

    if (ret == 0) {
        params.n_ctx = 1024;
        llama_context * ctx_pages = llama_init_from_model(model, common_context_params_to_llama(params));
        if (ctx_pages == nullptr) {
            fprintf(stderr, "%s : failed to init the 4-page context\n", __func__);
            ret = 1;
        } else {
            ret = test_concurrent_decode_pages(ctx_pages, batch);
            if (ret == 0) {
                ret = test_prefill_takes_empty_pages(ctx_pages, batch);
            }
            if (ret == 0) {
                ret = test_full_cache_uses_holes(ctx_pages, batch);
            }
            if (ret == 0) {
                ret = test_restore_takes_empty_pages(ctx_pages, batch);
            }
            if (ret == 0) {
                ret = test_decode_after_park_stays_in_page(ctx_pages, batch);
            }
            if (ret == 0) {
                ret = test_full_tail_reuses_owned_page(ctx_pages, batch);
            }
            llama_free(ctx_pages);
        }
    }

    llama_batch_free(batch);

    return ret;
}
