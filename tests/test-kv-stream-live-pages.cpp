#include "llama-kv-cache.h"
#include "testing.h"

#include <bitset>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

constexpr uint32_t PAGE_TOKENS = 256;

void fill(llama_kv_cells & cells, uint32_t begin, uint32_t end, llama_seq_id seq_id) {
    for (uint32_t i = begin; i < end; ++i) {
        cells.pos_set(i, llama_pos(i));
        cells.seq_add(i, seq_id);
    }
}

// one char per page
std::string live_pages(const llama_kv_cells & cells, std::initializer_list<llama_seq_id> seq_ids, uint32_t n_kv) {
    static std::vector<uint8_t> out;

    std::bitset<LLAMA_MAX_SEQ> seqs;
    for (const auto seq_id : seq_ids) {
        seqs.set(seq_id);
    }

    llama_kv_cache_live_pages(cells, seqs, n_kv, PAGE_TOKENS, out);

    std::string pages;
    for (const auto live : out) {
        pages += live ? '1' : '0';
    }
    return pages;
}

} // namespace

int main() {
    testing t;

    t.test("contiguous single sequence", [](testing & t) {
        llama_kv_cells cells;
        cells.resize(1024);
        fill(cells, 0, 1024, 0);

        t.assert_equal("every page is live", std::string("1111"), live_pages(cells, {0}, 1024));
        t.assert_equal("no page is live for an absent sequence", std::string("0000"), live_pages(cells, {1}, 1024));
    });

    t.test("disjoint sequences isolation", [](testing & t) {
        llama_kv_cells cells;
        cells.resize(53248);
        fill(cells, 0, 49152, 0);
        fill(cells, 49152, 53248, 1);

        const std::string seq0 = std::string(192, '1') + std::string(16, '0');
        const std::string seq1 = std::string(192, '0') + std::string(16, '1');

        t.assert_equal("seq 1 skips seq 0 pages", seq1, live_pages(cells, {1}, 53248));
        t.assert_equal("seq 0 skips seq 1 pages", seq0, live_pages(cells, {0}, 53248));
        t.assert_equal("mixed batch keeps every page", std::string(208, '1'), live_pages(cells, {0, 1}, 53248));
    });

    t.test("empty prefix hole", [](testing & t) {
        llama_kv_cells cells;
        cells.resize(53248);
        fill(cells, 49152, 53248, 1);

        const std::string seq1 = std::string(192, '0') + std::string(16, '1');

        t.assert_equal("empty prefix pages are skipped", seq1, live_pages(cells, {1}, 53248));
    });

    t.test("seq_cp shared cells", [](testing & t) {
        llama_kv_cells cells;
        cells.resize(512);
        cells.pos_set(100, 0);
        cells.seq_add(100, 0);
        cells.seq_add(100, 1);
        cells.pos_set(300, 1);
        cells.seq_add(300, 0);

        t.assert_equal("shared cell is live for seq 1", std::string("10"), live_pages(cells, {1}, 512));
        t.assert_equal("both pages are live for seq 0",  std::string("11"), live_pages(cells, {0}, 512));
        t.assert_equal("nothing is live for seq 2",      std::string("00"), live_pages(cells, {2}, 512));
    });

    t.test("partially used tail page", [](testing & t) {
        llama_kv_cells cells;
        cells.resize(512);
        fill(cells, 0, 257, 0);

        t.assert_equal("one cell keeps the tail page live", std::string("11"), live_pages(cells, {0}, 512));
    });

    t.test("interior hole and cells above n_kv", [](testing & t) {
        llama_kv_cells cells;
        cells.resize(2048);
        fill(cells, 0, 256, 0);
        fill(cells, 512, 768, 0);
        fill(cells, 1024, 2048, 0);

        t.assert_equal("hole pages are skipped and n_kv bounds the output", std::string("1010"), live_pages(cells, {0}, 1024));
    });

    t.test("live cell at the last index of a page", [](testing & t) {
        llama_kv_cells cells;
        cells.resize(768);
        fill(cells, 511, 512, 0);

        t.assert_equal("last cell keeps its page live", std::string("010"), live_pages(cells, {0}, 768));
    });

    return t.summary();
}
