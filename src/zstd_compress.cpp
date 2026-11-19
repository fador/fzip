// fzip — zstd compressor implementation (RFC 8878).
//
// Multi-tier LZ77 + repeat-offset-aware match finding (fast single-probe hash,
// double-hash, dual-hash chained lazy/lazy2, and BT4 binary search tree with
// multi-pass forward DP), Huffman literals (1-stream / 4-stream / treeless),
// FSE sequences (predefined / RLE / repeat / inline KL-normalized FSE), block
// RLE detection, and adaptive MDL block splitting.
#include "zstd.hpp"
#include "zstd_internal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>

#include "zstd_fse.hpp"
#include "zstd_huffman.hpp"
#include "zstd_predefined.hpp"
#include "zstd_sequence.hpp"

namespace fzip::zstd {

namespace {

constexpr int kMinMatch = 3;
constexpr int kMaxMatch = 131074;  // zstd max match length
constexpr int kBlockSize = 131072; // 128 KiB max zstd block uncompressed size

struct Match {
    int distance = 0;
    int length = 0;
};

struct Seq {
    int lit_len = 0;
    int match_len = 0;
    int match_dist = 0;
};

struct Code {
    int code = 0;
    int extra_bits = 0;
    int extra_val = 0;
};

// --------------------------------------------------------------------------
// Fast byte loads & hashing helpers
// --------------------------------------------------------------------------
inline auto load_u16(const std::uint8_t* p) -> std::uint16_t {
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

inline auto load_u32(const std::uint8_t* p) -> std::uint32_t {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline auto hash3(const std::uint8_t* p) -> std::uint32_t {
    const std::uint32_t h = (static_cast<std::uint32_t>(p[0]) << 16) |
                            (static_cast<std::uint32_t>(p[1]) << 8) |
                            static_cast<std::uint32_t>(p[2]);
    return (h * 2654435761u) >> 16;  // 16-bit
}

inline auto hash4(const std::uint8_t* p, int bits) -> std::uint32_t {
    return (load_u32(p) * 2654435761u) >> (32 - bits);
}

inline auto hash6(const std::uint8_t* p, int bits) -> std::uint32_t {
    const std::uint64_t v = static_cast<std::uint64_t>(load_u32(p)) |
                            (static_cast<std::uint64_t>(load_u16(p + 4)) << 32);
    return static_cast<std::uint32_t>((v * 0xCF1BBCDCB7A56463ULL) >> (64 - bits));
}

inline auto match_len_from(const std::uint8_t* a, const std::uint8_t* b,
                           int start_l, int maxl) -> int {
    int l = start_l;
    while (l + 8 <= maxl) {
        std::uint64_t va, vb;
        std::memcpy(&va, a + l, 8);
        std::memcpy(&vb, b + l, 8);
        if (va != vb) {
            if constexpr (std::endian::native == std::endian::little) {
                return l + (std::countr_zero(va ^ vb) >> 3);
            } else {
                return l + (std::countl_zero(va ^ vb) >> 3);
            }
        }
        l += 8;
    }
    while (l < maxl && a[l] == b[l]) ++l;
    return l;
}

inline auto is_uniform_bytes(const std::uint8_t* p, std::size_t n) -> bool {
    if (n == 0) return false;
    const std::uint8_t b0 = p[0];
    const std::uint64_t pat = static_cast<std::uint64_t>(b0) * 0x0101010101010101ULL;
    std::size_t i = 0;
    while (i + 8 <= n) {
        std::uint64_t v;
        std::memcpy(&v, p + i, 8);
        if (v != pat) return false;
        i += 8;
    }
    while (i < n) {
        if (p[i] != b0) return false;
        ++i;
    }
    return true;
}

// --------------------------------------------------------------------------
// O(1) RFC 8878 §4.2.2 Code LUTs
// --------------------------------------------------------------------------
constexpr auto init_ll_lut() -> std::array<Code, 64> {
    std::array<Code, 64> lut{};
    for (int i = 0; i < 16; ++i) lut[static_cast<std::size_t>(i)] = {i, 0, 0};
    struct Row { int min; int max; int code; int extra; int base; };
    constexpr Row rows[] = {
        {16, 17, 16, 1, 16}, {18, 19, 17, 1, 18}, {20, 21, 18, 1, 20},
        {22, 23, 19, 1, 22}, {24, 27, 20, 2, 24}, {28, 31, 21, 2, 28},
        {32, 39, 22, 3, 32}, {40, 47, 23, 3, 40}, {48, 63, 24, 4, 48},
    };
    for (const auto& r : rows) {
        for (int l = r.min; l <= r.max; ++l) {
            lut[static_cast<std::size_t>(l)] = {r.code, r.extra, l - r.base};
        }
    }
    return lut;
}

constexpr auto init_ml_lut() -> std::array<Code, 131> {
    std::array<Code, 131> lut{};
    for (int i = 3; i <= 34; ++i) lut[static_cast<std::size_t>(i)] = {i - 3, 0, 0};
    struct Row { int min; int max; int code; int extra; int base; };
    constexpr Row rows[] = {
        {35, 36, 32, 1, 35}, {37, 38, 33, 1, 37}, {39, 40, 34, 1, 39},
        {41, 42, 35, 1, 41}, {43, 46, 36, 2, 43}, {47, 50, 37, 2, 47},
        {51, 58, 38, 3, 51}, {59, 66, 39, 3, 59}, {67, 82, 40, 4, 67},
        {83, 98, 41, 4, 83}, {99, 130, 42, 5, 99},
    };
    for (const auto& r : rows) {
        for (int l = r.min; l <= r.max; ++l) {
            lut[static_cast<std::size_t>(l)] = {r.code, r.extra, l - r.base};
        }
    }
    return lut;
}

constexpr auto kLLLut = init_ll_lut();
constexpr auto kMLLut = init_ml_lut();

inline auto lit_len_code(int len) -> Code {
    if (len < 64) return kLLLut[static_cast<std::size_t>(len)];
    int hb = std::bit_width(static_cast<std::uint32_t>(len)) - 1;
    if (hb > 16) hb = 16;
    const int base = 1 << hb;
    return {19 + hb, hb, len - base};
}

inline auto match_len_code(int len) -> Code {
    if (len < 131) return kMLLut[static_cast<std::size_t>(len)];
    const int v = len - 3;
    int hb = std::bit_width(static_cast<std::uint32_t>(v)) - 1;
    if (hb > 16) hb = 16;
    const int base = (1 << hb) + 3;
    return {36 + hb, hb, len - base};
}

inline auto offset_code(int dist) -> Code {
    const int v = dist + 3;
    int code = std::bit_width(static_cast<std::uint32_t>(v)) - 1;
    if (code < 2) code = 2;
    return {code, code, v - (1 << code)};
}

// Pick the cheapest offset code for `dist` given the current repeat state,
// updating `rep` to match ZSTD_decodeSequence's repeat-offset state machine.
inline auto choose_offset(int dist, int lit_len, RepeatOffsets& rep) -> Code {
    const bool ll0 = (lit_len == 0);
    if (!ll0) {
        if (dist == rep.offsets[0]) {
            return {0, 0, 0};
        }
        if (dist == rep.offsets[1]) {
            const int r = rep.offsets[1];
            rep.offsets[1] = rep.offsets[0];
            rep.offsets[0] = r;
            return {1, 1, 0};
        }
        if (dist == rep.offsets[2]) {
            const int r = rep.offsets[2];
            rep.offsets[2] = rep.offsets[1];
            rep.offsets[1] = rep.offsets[0];
            rep.offsets[0] = r;
            return {1, 1, 1};
        }
    } else {
        if (dist == rep.offsets[1]) {
            const int r = rep.offsets[1];
            rep.offsets[1] = rep.offsets[0];
            rep.offsets[0] = r;
            return {0, 0, 0};
        }
        if (dist == rep.offsets[2]) {
            const int r = rep.offsets[2];
            rep.offsets[2] = rep.offsets[1];
            rep.offsets[1] = rep.offsets[0];
            rep.offsets[0] = r;
            return {1, 1, 0};
        }
        if (rep.offsets[0] > 1 && dist == rep.offsets[0] - 1) {
            const int newv = rep.offsets[0] - 1;
            rep.offsets[2] = rep.offsets[1];
            rep.offsets[1] = rep.offsets[0];
            rep.offsets[0] = newv;
            return {1, 1, 1};
        }
    }
    const Code c = offset_code(dist);
    rep.offsets[2] = rep.offsets[1];
    rep.offsets[1] = rep.offsets[0];
    rep.offsets[0] = dist;
    return c;
}

// --------------------------------------------------------------------------
// Thread-local epoch-reset workspace (clamped to actual input window).
// --------------------------------------------------------------------------
constexpr int kHashSize3 = 1 << 16;
constexpr int kHashSize4Max = 1 << 18;
constexpr int kHashSize6 = 1 << 17;

struct ZstdWorkspace {
    std::vector<std::int32_t> head3;
    std::vector<std::int32_t> head4;
    std::vector<std::int32_t> head6;
    std::vector<std::int32_t> prev3;
    std::vector<std::int32_t> prev4;
    std::vector<std::int32_t> bt_left;
    std::vector<std::int32_t> bt_right;
    std::int32_t epoch_base = 1;

    auto begin_frame(std::size_t total_size, int actual_window,
                     bool need_h3, bool need_h4_chain, bool need_h6,
                     bool need_bt) -> std::int32_t {
        if (head4.empty()) head4.assign(kHashSize4Max, 0);
        if (need_h3 && head3.empty()) head3.assign(kHashSize3, 0);
        if (need_h6 && head6.empty()) head6.assign(kHashSize6, 0);

        const std::size_t wsz = static_cast<std::size_t>(actual_window);
        if (need_h3 && prev3.size() < wsz) prev3.resize(wsz);
        if (need_h4_chain && prev4.size() < wsz) prev4.resize(wsz);
        if (need_bt) {
            if (bt_left.size() < wsz) bt_left.resize(wsz);
            if (bt_right.size() < wsz) bt_right.resize(wsz);
        }

        if (epoch_base + static_cast<std::int64_t>(total_size) + 64 >= 0x3FFFFFFFLL) {
            if (!head3.empty()) std::fill(head3.begin(), head3.end(), 0);
            std::fill(head4.begin(), head4.end(), 0);
            if (!head6.empty()) std::fill(head6.begin(), head6.end(), 0);
            epoch_base = 1;
        }
        const std::int32_t base = epoch_base;
        epoch_base += static_cast<std::int32_t>(total_size) + 1;
        return base;
    }
};

auto get_zstd_workspace() -> ZstdWorkspace& {
    thread_local std::unique_ptr<ZstdWorkspace> ws;
    if (!ws) ws = std::make_unique<ZstdWorkspace>();
    return *ws;
}

// Shared match scoring for Tiers 1-3 (approx net savings in bits).
inline auto match_savings_bits(int len, int dist, bool is_rep) -> int {
    if (len < kMinMatch) return -1000;
    const int off_bits = is_rep ? 2 : (offset_code(dist).code + 4);
    return len * 7 - off_bits;
}

struct RepMatch {
    int dist = 0;
    int len = 0;
};

inline auto find_rep_match(const std::uint8_t* data, std::size_t pos,
                           std::size_t end, int max_dist,
                           const RepeatOffsets& rep) -> RepMatch {
    RepMatch best;
    const int maxl = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(kMaxMatch), end - pos));
    if (maxl < kMinMatch) return best;
    for (int idx = 0; idx < 3; ++idx) {
        const int d = rep.offsets[idx];
        if (d <= 0 || d > max_dist || static_cast<std::size_t>(d) > pos) continue;
        const std::size_t cand = pos - static_cast<std::size_t>(d);
        if (data[cand] != data[pos] || data[cand + 1] != data[pos + 1] ||
            data[cand + 2] != data[pos + 2]) {
            continue;
        }
        const int l = match_len_from(data + cand, data + pos, 3, maxl);
        if (l > best.len) {
            best.dist = d;
            best.len = l;
        }
    }
    return best;
}

// --------------------------------------------------------------------------
// Tier 1 (levels 1-2): Fast 17-bit 4-byte hash + short probe + head3 + rep
// --------------------------------------------------------------------------
void match_tier1_block(const std::uint8_t* data, std::size_t start,
                       std::size_t end, ZstdWorkspace& ws, std::int32_t base,
                       int actual_window, int effort,
                       std::vector<Seq>& seqs,
                       std::vector<std::uint8_t>& literals,
                       RepeatOffsets& rep) {
    seqs.clear();
    literals.clear();
    constexpr int kH4Bits = 17;
    const int max_dist = actual_window - 1;
    const std::size_t mask = static_cast<std::size_t>(actual_window - 1);

    auto insert_pos = [&](std::size_t p) {
        if (p + kMinMatch <= end) {
            const std::uint32_t h3 = hash3(data + p);
            ws.prev3[p & mask] = ws.head3[h3];
            ws.head3[h3] = base + static_cast<std::int32_t>(p);
            if (p + 4 <= end) {
                const std::uint32_t h4 = hash4(data + p, kH4Bits);
                ws.prev4[p & mask] = ws.head4[h4];
                ws.head4[h4] = base + static_cast<std::int32_t>(p);
            }
        }
    };

    std::size_t inserted_up_to = start;
    auto ensure_inserted = [&](std::size_t target) {
        while (inserted_up_to < target) insert_pos(inserted_up_to++);
    };

    std::size_t pos = start;
    std::size_t lit_start = start;
    while (pos < end) {
        ensure_inserted(pos);
        const int maxl = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(kMaxMatch), end - pos));
        const std::size_t min_p =
            (pos > static_cast<std::size_t>(max_dist))
                ? (pos - static_cast<std::size_t>(max_dist))
                : 0;
        const std::int32_t limit_tag = base + static_cast<std::int32_t>(min_p);

        const RepMatch rm = find_rep_match(data, pos, end, max_dist, rep);
        Match m;
        if (pos + 4 <= end) {
            const std::uint32_t p4 = load_u32(data + pos);
            std::int32_t cand_tag = ws.head4[hash4(data + pos, kH4Bits)];
            int tries = effort;
            while (cand_tag >= limit_tag && tries-- > 0) {
                const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
                if (load_u32(data + cand) == p4) {
                    const int l = match_len_from(data + cand, data + pos, 4, maxl);
                    if (l > m.length) {
                        m.length = l;
                        m.distance = static_cast<int>(pos - cand);
                        if (l >= 32) break;
                    }
                }
                cand_tag = ws.prev4[cand & mask];
            }
        }

        if (m.length < 4 && pos + 3 <= end) {
            const std::int32_t cand_tag = ws.head3[hash3(data + pos)];
            if (cand_tag >= limit_tag) {
                const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
                const int d = static_cast<int>(pos - cand);
                if (d <= 4096 && data[cand] == data[pos] &&
                    data[cand + 1] == data[pos + 1] &&
                    data[cand + 2] == data[pos + 2]) {
                    m.length = match_len_from(data + cand, data + pos, 3, maxl);
                    m.distance = d;
                }
            }
        }

        bool use_rep = (rm.len >= kMinMatch);
        if (use_rep && m.length >= 4 &&
            match_savings_bits(m.length, m.distance, false) >
                match_savings_bits(rm.len, rm.dist, true)) {
            use_rep = false;
        }
        if (use_rep) {
            m.length = rm.len;
            m.distance = rm.dist;
        }

        if (m.length < kMinMatch) {
            ++pos;
            continue;
        }

        if (pos > lit_start) {
            literals.insert(literals.end(), data + lit_start, data + pos);
        }
        Seq s;
        s.lit_len = static_cast<int>(pos - lit_start);
        s.match_len = m.length;
        s.match_dist = m.distance;
        choose_offset(s.match_dist, s.lit_len, rep);
        seqs.push_back(s);

        const std::size_t next_pos = pos + static_cast<std::size_t>(m.length);
        if (m.length >= 6) {
            insert_pos(pos);
            if (next_pos > pos + 2) insert_pos(next_pos - 1);
            inserted_up_to = next_pos;
        } else {
            ensure_inserted(next_pos);
        }
        pos = next_pos;
        lit_start = pos;
    }
    if (lit_start < end) {
        literals.insert(literals.end(), data + lit_start, data + end);
    }
    ensure_inserted(end);
}

// --------------------------------------------------------------------------
// Tier 2 (levels 3-5): Double-hash (6-byte + short 4-byte chain + head3)
// --------------------------------------------------------------------------
void match_tier2_block(const std::uint8_t* data, std::size_t start,
                       std::size_t end, ZstdWorkspace& ws, std::int32_t base,
                       int actual_window, int effort, int nice_len,
                       std::vector<Seq>& seqs,
                       std::vector<std::uint8_t>& literals,
                       RepeatOffsets& rep) {
    seqs.clear();
    literals.clear();
    constexpr int kH4Bits = 17;
    constexpr int kH6Bits = 17;
    const int max_dist = actual_window - 1;
    const std::size_t mask = static_cast<std::size_t>(actual_window - 1);

    auto insert_pos = [&](std::size_t p) {
        if (p + 3 <= end) {
            ws.head3[hash3(data + p)] = base + static_cast<std::int32_t>(p);
        }
        if (p + 4 <= end) {
            const std::uint32_t h4 = hash4(data + p, kH4Bits);
            ws.prev4[p & mask] = ws.head4[h4];
            ws.head4[h4] = base + static_cast<std::int32_t>(p);
            if (p + 6 <= end) {
                ws.head6[hash6(data + p, kH6Bits)] = base + static_cast<std::int32_t>(p);
            }
        }
    };

    std::size_t inserted_up_to = start;
    auto ensure_inserted = [&](std::size_t target) {
        while (inserted_up_to < target) insert_pos(inserted_up_to++);
    };

    std::size_t pos = start;
    std::size_t lit_start = start;
    while (pos < end) {
        ensure_inserted(pos);
        const int maxl = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(kMaxMatch), end - pos));
        const int eff_nice = std::min(nice_len, maxl);
        const std::size_t min_p =
            (pos > static_cast<std::size_t>(max_dist))
                ? (pos - static_cast<std::size_t>(max_dist))
                : 0;
        const std::int32_t limit_tag = base + static_cast<std::int32_t>(min_p);

        const RepMatch rm = find_rep_match(data, pos, end, max_dist, rep);
        Match m;

        // 1. Check 6-byte hash table first.
        if (pos + 6 <= end) {
            const std::int32_t cand_tag = ws.head6[hash6(data + pos, kH6Bits)];
            if (cand_tag >= limit_tag) {
                const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
                if (load_u32(data + cand) == load_u32(data + pos) &&
                    load_u16(data + cand + 4) == load_u16(data + pos + 4)) {
                    m.length = match_len_from(data + cand, data + pos, 6, maxl);
                    m.distance = static_cast<int>(pos - cand);
                }
            }
        }

        // 2. Search short 4-byte chain if not yet at nice_len.
        if (m.length < eff_nice && pos + 4 <= end) {
            const std::uint32_t p4 = load_u32(data + pos);
            std::int32_t cand_tag = ws.head4[hash4(data + pos, kH4Bits)];
            int tries = effort;
            while (cand_tag >= limit_tag && tries-- > 0) {
                const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
                if (m.length >= 4 && m.length < maxl &&
                    (data[cand + static_cast<std::size_t>(m.length)] !=
                         data[pos + static_cast<std::size_t>(m.length)] ||
                     data[cand + static_cast<std::size_t>(m.length) - 1] !=
                         data[pos + static_cast<std::size_t>(m.length) - 1])) {
                    cand_tag = ws.prev4[cand & mask];
                    continue;
                }
                if (load_u32(data + cand) != p4) {
                    cand_tag = ws.prev4[cand & mask];
                    continue;
                }
                const int l = match_len_from(data + cand, data + pos, 4, maxl);
                if (l > m.length) {
                    m.length = l;
                    m.distance = static_cast<int>(pos - cand);
                    if (l >= eff_nice) break;
                }
                cand_tag = ws.prev4[cand & mask];
            }
        }

        // 3. Fallback 1-probe 3-byte check for close 3-byte matches.
        if (m.length < 4 && pos + 3 <= end) {
            const std::int32_t cand_tag = ws.head3[hash3(data + pos)];
            if (cand_tag >= limit_tag) {
                const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
                const int d = static_cast<int>(pos - cand);
                if (d <= 4096 && data[cand] == data[pos] &&
                    data[cand + 1] == data[pos + 1] &&
                    data[cand + 2] == data[pos + 2]) {
                    m.length = match_len_from(data + cand, data + pos, 3, maxl);
                    m.distance = d;
                }
            }
        }

        bool use_rep = (rm.len >= kMinMatch);
        if (use_rep && m.length >= 4 &&
            match_savings_bits(m.length, m.distance, false) >
                match_savings_bits(rm.len, rm.dist, true)) {
            use_rep = false;
        }
        if (use_rep) {
            m.length = rm.len;
            m.distance = rm.dist;
        }

        if (m.length < kMinMatch) {
            ++pos;
            continue;
        }

        if (pos > lit_start) {
            literals.insert(literals.end(), data + lit_start, data + pos);
        }
        Seq s;
        s.lit_len = static_cast<int>(pos - lit_start);
        s.match_len = m.length;
        s.match_dist = m.distance;
        choose_offset(s.match_dist, s.lit_len, rep);
        seqs.push_back(s);

        const std::size_t next_pos = pos + static_cast<std::size_t>(m.length);
        if (m.length >= 16) {
            insert_pos(pos);
            if (next_pos > pos + 2) insert_pos(next_pos - 1);
            inserted_up_to = next_pos;
        } else {
            ensure_inserted(next_pos);
        }
        pos = next_pos;
        lit_start = pos;
    }
    if (lit_start < end) {
        literals.insert(literals.end(), data + lit_start, data + end);
    }
    ensure_inserted(end);
}

// --------------------------------------------------------------------------
// Tier 3 (levels 6-15): Dual-hash chained lazy / lazy2 + repeat offsets
// --------------------------------------------------------------------------
void match_tier3_block(const std::uint8_t* data, std::size_t start,
                       std::size_t end, ZstdWorkspace& ws, std::int32_t base,
                       int actual_window, int effort, int nice_len,
                       int lazy_depth, std::vector<Seq>& seqs,
                       std::vector<std::uint8_t>& literals,
                       RepeatOffsets& rep) {
    seqs.clear();
    literals.clear();
    constexpr int kH4Bits = 17;
    constexpr int kH6Bits = 17;
    const int max_dist = actual_window - 1;
    const std::size_t mask = static_cast<std::size_t>(actual_window - 1);

    auto insert_pos = [&](std::size_t p) {
        if (p + kMinMatch <= end) {
            const std::uint32_t h3 = hash3(data + p);
            ws.prev3[p & mask] = ws.head3[h3];
            ws.head3[h3] = base + static_cast<std::int32_t>(p);
            if (p + 4 <= end) {
                const std::uint32_t h4 = hash4(data + p, kH4Bits);
                ws.prev4[p & mask] = ws.head4[h4];
                ws.head4[h4] = base + static_cast<std::int32_t>(p);
                if (p + 6 <= end) {
                    ws.head6[hash6(data + p, kH6Bits)] = base + static_cast<std::int32_t>(p);
                }
            }
        }
    };

    std::size_t inserted_up_to = start;
    auto ensure_inserted = [&](std::size_t target) {
        while (inserted_up_to < target) insert_pos(inserted_up_to++);
    };

    auto find_hc = [&](std::size_t pos, int max_tries, Match& best,
                       bool& is_rep) {
        best = {};
        is_rep = false;
        if (pos + kMinMatch > end) return;
        const int maxl = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(kMaxMatch), end - pos));
        const int eff_nice = std::min(nice_len, maxl);
        const std::size_t min_p =
            (pos > static_cast<std::size_t>(max_dist))
                ? (pos - static_cast<std::size_t>(max_dist))
                : 0;
        const std::int32_t limit_tag = base + static_cast<std::int32_t>(min_p);

        const RepMatch rm = find_rep_match(data, pos, end, max_dist, rep);
        if (rm.len >= eff_nice) {
            best = {rm.dist, rm.len};
            is_rep = true;
            return;
        }

        Match m;
        if (pos + 6 <= end) {
            const std::int32_t cand_tag = ws.head6[hash6(data + pos, kH6Bits)];
            if (cand_tag >= limit_tag) {
                const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
                if (load_u32(data + cand) == load_u32(data + pos) &&
                    load_u16(data + cand + 4) == load_u16(data + pos + 4)) {
                    m.length = match_len_from(data + cand, data + pos, 6, maxl);
                    m.distance = static_cast<int>(pos - cand);
                }
            }
        }

        if (m.length < eff_nice && pos + 4 <= end) {
            const std::uint32_t p4 = load_u32(data + pos);
            std::int32_t cand_tag = ws.head4[hash4(data + pos, kH4Bits)];
            int tries = max_tries;
            while (cand_tag >= limit_tag && tries-- > 0) {
                const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
                if (m.length >= 4 && m.length < maxl &&
                    (data[cand + static_cast<std::size_t>(m.length)] !=
                         data[pos + static_cast<std::size_t>(m.length)] ||
                     data[cand + static_cast<std::size_t>(m.length) - 1] !=
                         data[pos + static_cast<std::size_t>(m.length) - 1])) {
                    cand_tag = ws.prev4[cand & mask];
                    continue;
                }
                if (load_u32(data + cand) != p4) {
                    cand_tag = ws.prev4[cand & mask];
                    continue;
                }
                const int l = match_len_from(data + cand, data + pos, 4, maxl);
                const int d = static_cast<int>(pos - cand);
                if (l > m.length) {
                    m.length = l;
                    m.distance = d;
                    if (l >= eff_nice) break;
                }
                cand_tag = ws.prev4[cand & mask];
            }
        }

        if (m.length < 4 || (m.length == 4 && m.distance > 32768)) {
            std::int32_t cand_tag = ws.head3[hash3(data + pos)];
            int tries = std::min(max_tries, 16);
            const std::uint8_t b0 = data[pos];
            const std::uint8_t b1 = data[pos + 1];
            const std::uint8_t b2 = data[pos + 2];
            while (cand_tag >= limit_tag && tries-- > 0) {
                const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
                const int d = static_cast<int>(pos - cand);
                if (d > 16384 || (m.length == 4 && d >= 512)) break;
                if (data[cand] == b0 && data[cand + 1] == b1 && data[cand + 2] == b2) {
                    const int l = match_len_from(data + cand, data + pos, 3, maxl);
                    if (m.length < 4 || l > m.length ||
                        (l == 3 && d < 512 && m.distance > 32768)) {
                        m.length = l;
                        m.distance = d;
                    }
                    break;
                }
                cand_tag = ws.prev3[cand & mask];
            }
        }

        if (rm.len >= kMinMatch &&
            match_savings_bits(rm.len, rm.dist, true) >=
                match_savings_bits(m.length, m.distance, false)) {
            best = {rm.dist, rm.len};
            is_rep = true;
        } else {
            best = m;
            is_rep = false;
        }
    };

    std::size_t pos = start;
    std::size_t lit_start = start;
    const int good_len = std::max(16, nice_len >> 2);
    while (pos < end) {
        ensure_inserted(pos);
        Match m;
        bool m_rep = false;
        find_hc(pos, effort, m, m_rep);
        if (m.length < kMinMatch) {
            ++pos;
            continue;
        }

        while (pos + 1 < end && m.length < nice_len) {
            const int cur_score = match_savings_bits(m.length, m.distance, m_rep);
            const int eff = (m.length >= good_len) ? std::max(8, effort >> 2) : effort;
            ensure_inserted(pos + 1);
            Match m2;
            bool m2_rep = false;
            find_hc(pos + 1, eff, m2, m2_rep);
            if (match_savings_bits(m2.length, m2.distance, m2_rep) > cur_score + 1) {
                ++pos;
                m = m2;
                m_rep = m2_rep;
                continue;
            }
            if (lazy_depth >= 2 && pos + 2 < end && m.length < (nice_len >> 1)) {
                ensure_inserted(pos + 2);
                Match m3;
                bool m3_rep = false;
                find_hc(pos + 2, eff, m3, m3_rep);
                if (match_savings_bits(m3.length, m3.distance, m3_rep) > cur_score + 8) {
                    pos += 2;
                    m = m3;
                    m_rep = m3_rep;
                    continue;
                }
            }
            break;
        }

        if (pos > lit_start) {
            literals.insert(literals.end(), data + lit_start, data + pos);
        }
        Seq s;
        s.lit_len = static_cast<int>(pos - lit_start);
        s.match_len = m.length;
        s.match_dist = m.distance;
        choose_offset(s.match_dist, s.lit_len, rep);
        seqs.push_back(s);

        const std::size_t next_pos = pos + static_cast<std::size_t>(m.length);
        ensure_inserted(next_pos);
        pos = next_pos;
        lit_start = pos;
    }
    if (lit_start < end) {
        literals.insert(literals.end(), data + lit_start, data + end);
    }
    ensure_inserted(end);
}

// --------------------------------------------------------------------------
// Tier 4 (levels 16-22): BT4 Binary Search Tree + Multi-Pass Forward DP
// --------------------------------------------------------------------------
constexpr int kMaxZstdCands = 32;

inline void bt_skip_insert(const std::uint8_t* data, std::size_t end,
                           std::size_t pos, ZstdWorkspace& ws,
                           std::int32_t base, std::size_t mask,
                           std::size_t min_p, int depth, int cmp_cap) {
    if (pos + kMinMatch <= end) {
        const std::uint32_t h3 = hash3(data + pos);
        ws.prev3[pos & mask] = ws.head3[h3];
        ws.head3[h3] = base + static_cast<std::int32_t>(pos);
    }
    if (pos + 4 > end) return;

    const std::int32_t limit_tag = base + static_cast<std::int32_t>(min_p);
    const int maxl = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(cmp_cap), end - pos));
    const std::uint32_t h4 = hash4(data + pos, 18);
    std::int32_t cur_tag = ws.head4[h4];
    ws.head4[h4] = base + static_cast<std::int32_t>(pos);

    const std::size_t pos_slot = pos & mask;
    std::int32_t* ptr_left = &ws.bt_left[pos_slot];
    std::int32_t* ptr_right = &ws.bt_right[pos_slot];
    int len_left = 0;
    int len_right = 0;
    int tries = depth;
    while (cur_tag >= limit_tag && tries-- > 0) {
        const std::size_t cand = static_cast<std::size_t>(cur_tag - base);
        const std::size_t cand_slot = cand & mask;
        int len = std::min(len_left, len_right);
        len = match_len_from(data + cand, data + pos, len, maxl);
        if (len >= maxl) {
            *ptr_left = ws.bt_left[cand_slot];
            *ptr_right = ws.bt_right[cand_slot];
            return;
        }
        if (data[cand + static_cast<std::size_t>(len)] <
            data[pos + static_cast<std::size_t>(len)]) {
            *ptr_left = cur_tag;
            ptr_left = &ws.bt_right[cand_slot];
            cur_tag = *ptr_left;
            len_left = len;
        } else {
            *ptr_right = cur_tag;
            ptr_right = &ws.bt_left[cand_slot];
            cur_tag = *ptr_right;
            len_right = len;
        }
    }
    *ptr_left = 0;
    *ptr_right = 0;
}

auto bt_find_and_insert(const std::uint8_t* data, std::size_t end,
                        std::size_t pos, ZstdWorkspace& ws,
                        std::int32_t base, std::size_t mask,
                        std::size_t min_p, int depth, int nice_len,
                        Match (&out)[kMaxZstdCands]) -> int {
    int n = 0;
    if (pos + kMinMatch > end) return 0;

    const int maxl = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(kMaxMatch), end - pos));
    const std::int32_t limit_tag = base + static_cast<std::int32_t>(min_p);

    if (pos + 4 <= end) {
        const int eff_nice = std::min(nice_len, maxl);
        const std::uint32_t h4 = hash4(data + pos, 18);
        std::int32_t cur_tag = ws.head4[h4];
        ws.head4[h4] = base + static_cast<std::int32_t>(pos);

        const std::size_t pos_slot = pos & mask;
        std::int32_t* ptr_left = &ws.bt_left[pos_slot];
        std::int32_t* ptr_right = &ws.bt_right[pos_slot];
        int len_left = 0;
        int len_right = 0;
        int best_len = 3;
        int tries = depth;
        bool attached = false;

        while (cur_tag >= limit_tag && tries-- > 0) {
            const std::size_t cand = static_cast<std::size_t>(cur_tag - base);
            const std::size_t cand_slot = cand & mask;
            int len = std::min(len_left, len_right);
            len = match_len_from(data + cand, data + pos, len, eff_nice);

            if (len > best_len) {
                int true_len = len;
                if (len == eff_nice && eff_nice < maxl) {
                    true_len = match_len_from(data + cand, data + pos, eff_nice, maxl);
                }
                const int d = static_cast<int>(pos - cand);
                if (n < kMaxZstdCands) {
                    out[n++] = Match{d, true_len};
                } else {
                    out[n - 1] = Match{d, true_len};
                }
                best_len = true_len;
                if (len >= eff_nice) {
                    *ptr_left = ws.bt_left[cand_slot];
                    *ptr_right = ws.bt_right[cand_slot];
                    attached = true;
                    break;
                }
            } else if (len >= eff_nice) {
                *ptr_left = ws.bt_left[cand_slot];
                *ptr_right = ws.bt_right[cand_slot];
                attached = true;
                break;
            }

            if (data[cand + static_cast<std::size_t>(len)] <
                data[pos + static_cast<std::size_t>(len)]) {
                *ptr_left = cur_tag;
                ptr_left = &ws.bt_right[cand_slot];
                cur_tag = *ptr_left;
                len_left = len;
            } else {
                *ptr_right = cur_tag;
                ptr_right = &ws.bt_left[cand_slot];
                cur_tag = *ptr_right;
                len_right = len;
            }
        }
        if (!attached) {
            *ptr_left = 0;
            *ptr_right = 0;
        }
    }

    // Closest 3+ byte match from head3 (all 4+ matches <= depth are already in BT4).
    {
        const std::uint32_t h3 = hash3(data + pos);
        std::int32_t cand_tag = ws.head3[h3];
        const int max_d3 = (n > 0) ? out[0].distance : static_cast<int>(mask + 1);
        if (max_d3 > 1) {
            int tries = 16;
            const std::uint8_t b0 = data[pos];
            const std::uint8_t b1 = data[pos + 1];
            const std::uint8_t b2 = data[pos + 2];
            while (cand_tag >= limit_tag && tries-- > 0) {
                const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
                const int d = static_cast<int>(pos - cand);
                if (d >= max_d3) break;
                if (data[cand] == b0 && data[cand + 1] == b1 && data[cand + 2] == b2) {
                    const int l = match_len_from(data + cand, data + pos, 3, maxl);
                    int w = 0;
                    for (int j = 0; j < n; ++j) {
                        if (!(out[j].distance >= d && out[j].length <= l)) {
                            out[w++] = out[j];
                        }
                    }
                    n = w;
                    if (n < kMaxZstdCands) {
                        for (int j = n; j > 0; --j) out[j] = out[j - 1];
                        out[0] = Match{d, l};
                        ++n;
                    } else {
                        for (int j = n - 1; j > 0; --j) out[j] = out[j - 1];
                        out[0] = Match{d, l};
                    }
                    break;
                }
                cand_tag = ws.prev3[cand & mask];
            }
        }
        ws.prev3[pos & mask] = ws.head3[h3];
        ws.head3[h3] = base + static_cast<std::int32_t>(pos);
    }

    return n;
}

// Fast entropy cost estimate for optimal pass selection.
auto estimate_block_total_bits(const std::vector<Seq>& seqs,
                               const std::vector<std::uint8_t>& lits,
                               const RepeatOffsets& initial_rep) -> double {
    const int n_lit = static_cast<int>(lits.size());
    double lit_bits = static_cast<double>(n_lit) * 8.0 + 24.0;
    if (n_lit >= 18) {
        std::uint32_t lf[256]{};
        int distinct = 0;
        for (std::uint8_t b : lits) {
            if (lf[b] == 0) ++distinct;
            lf[b]++;
        }
        if (distinct == 1) {
            lit_bits = 24.0;
        } else if (distinct >= 2) {
            HufTable ht;
            if (huf_literals_table_from_freqs(lf, ht)) {
                double hb = static_cast<double>(distinct) * 4.5 + 32.0;
                for (int s = 0; s < 256; ++s) {
                    if (lf[s] > 0) {
                        hb += static_cast<double>(lf[s]) *
                              static_cast<double>(ht.lengths[static_cast<std::size_t>(s)]);
                    }
                }
                if (hb < lit_bits) lit_bits = hb;
            }
        }
    }

    const int n_seq = static_cast<int>(seqs.size());
    if (n_seq == 0) return lit_bits + 8.0;

    int ll_freq[36]{};
    int of_freq[32]{};
    int ml_freq[53]{};
    double extra_bits = 0.0;
    RepeatOffsets r = initial_rep;
    for (const Seq& s : seqs) {
        const Code lc = lit_len_code(s.lit_len);
        const Code oc = choose_offset(s.match_dist, s.lit_len, r);
        const Code mc = match_len_code(s.match_len);
        ll_freq[lc.code]++;
        of_freq[oc.code]++;
        ml_freq[mc.code]++;
        extra_bits += static_cast<double>(lc.extra_bits + oc.extra_bits + mc.extra_bits);
    }

    auto stream_bits = [n_seq](const int* freq, int max_c) -> double {
        int present = 0;
        for (int c = 0; c <= max_c; ++c) {
            if (freq[c] > 0) ++present;
        }
        if (present <= 1) return 8.0;
        double bits = 0.0;
        const double log_tot = std::log2(static_cast<double>(n_seq));
        for (int c = 0; c <= max_c; ++c) {
            if (freq[c] > 0) {
                bits += static_cast<double>(freq[c]) *
                        (log_tot - std::log2(static_cast<double>(freq[c])));
            }
        }
        const double hdr = (n_seq >= 16) ? std::min(160.0, static_cast<double>(present) * 5.0 + 12.0) : 0.0;
        return bits + hdr;
    };

    return lit_bits + extra_bits +
           stream_bits(ll_freq, 35) +
           stream_bits(of_freq, 31) +
           stream_bits(ml_freq, 52);
}

inline auto sym_cost(int freq, int total, float cap) -> float {
    if (freq <= 0 || total <= 0) return cap;
    return std::min(cap, -std::log2(static_cast<float>(freq) / static_cast<float>(total)));
}

void build_sequences_optimal(const std::uint8_t* data, std::size_t start,
                             std::size_t end, ZstdWorkspace& ws,
                             std::int32_t base, int actual_window,
                             int bt_depth, int nice_len, int passes,
                             std::vector<Seq>& out_seqs,
                             std::vector<std::uint8_t>& literals,
                             RepeatOffsets& rep) {
    out_seqs.clear();
    literals.clear();
    const int N = static_cast<int>(end - start);
    if (N <= 0) return;

    const RepeatOffsets initial_rep = rep;
    const int max_dist = actual_window - 1;  // avoid BT4 slot aliasing
    const std::size_t mask = static_cast<std::size_t>(actual_window - 1);
    const int sufficient_len = nice_len;

    auto min_p_for = [&](std::size_t pos) -> std::size_t {
        return (pos > static_cast<std::size_t>(max_dist))
                   ? (pos - static_cast<std::size_t>(max_dist))
                   : 0;
    };

    // 1. Forward pass: collect Pareto-optimal matches with O(1) long-match propagation.
    std::vector<Match> pool;
    pool.reserve(static_cast<std::size_t>(N) * 2);
    std::vector<int> cand_start(static_cast<std::size_t>(N) + 1, 0);
    for (int p = 0; p < N; ++p) {
        const std::size_t pos = start + static_cast<std::size_t>(p);
        cand_start[static_cast<std::size_t>(p)] = static_cast<int>(pool.size());
        Match local[kMaxZstdCands];
        const int nc = bt_find_and_insert(data, end, pos, ws, base, mask,
                                          min_p_for(pos), bt_depth, nice_len,
                                          local);
        for (int j = 0; j < nc; ++j) pool.push_back(local[j]);

        if (nc > 0 && local[nc - 1].length >= sufficient_len) {
            const int long_len = local[nc - 1].length;
            const int long_dist = local[nc - 1].distance;
            const int skip_end = std::min(N - 1, p + long_len - kMinMatch);
            for (int q = p + 1; q <= skip_end; ++q) {
                const std::size_t qp = start + static_cast<std::size_t>(q);
                cand_start[static_cast<std::size_t>(q)] = static_cast<int>(pool.size());
                pool.push_back(Match{long_dist, long_len - (q - p)});
                bt_skip_insert(data, end, qp, ws, base, mask, min_p_for(qp),
                               8, 64);
            }
            p = skip_end;
        }
    }
    cand_start[static_cast<std::size_t>(N)] = static_cast<int>(pool.size());

    int ml_range_max[53];
    for (int c = 0; c < 53; ++c) {
        ml_range_max[c] = matchlen_code_to_base(c) +
                          (1 << matchlen_code_to_extra(c)) - 1;
    }

    // 2. Seed statistics with a repeat-aware greedy parse over `pool`.
    int lit_freq[256]{};
    int ll_freq[36]{};
    int of_freq[32]{};
    int ml_freq[53]{};
    int n_lit = 0;
    int n_seq = 0;
    {
        int p = 0;
        int lit_run = 0;
        RepeatOffsets greedy_rep = initial_rep;
        while (p < N) {
            const std::size_t pos = start + static_cast<std::size_t>(p);
            int L = 0;
            int d = 0;
            const int ci_begin = cand_start[static_cast<std::size_t>(p)];
            const int ci_end = cand_start[static_cast<std::size_t>(p) + 1];
            if (ci_end > ci_begin) {
                const Match& longest = pool[static_cast<std::size_t>(ci_end - 1)];
                if (!(longest.length == 3 && longest.distance > 16384)) {
                    L = longest.length;
                    d = longest.distance;
                }
            }
            const RepMatch rm = find_rep_match(data, pos, end, max_dist, greedy_rep);
            if (rm.len >= kMinMatch && rm.len + 2 >= L) {
                L = rm.len;
                d = rm.dist;
            }
            if (L >= kMinMatch) {
                ll_freq[lit_len_code(lit_run).code]++;
                const Code oc = choose_offset(d, lit_run, greedy_rep);
                of_freq[oc.code]++;
                ml_freq[match_len_code(L).code]++;
                ++n_seq;
                p += L;
                lit_run = 0;
            } else {
                lit_freq[data[pos]]++;
                ++n_lit;
                ++p;
                ++lit_run;
            }
        }
        for (const auto& c : pool) {
            const int mc = match_len_code(c.length).code;
            if (ml_freq[mc] == 0) { ml_freq[mc] = 1; ++n_seq; }
            const int oc = offset_code(c.distance).code;
            if (oc < 32 && of_freq[oc] == 0) { of_freq[oc] = 1; }
        }
        if (of_freq[0] == 0) of_freq[0] = 1;
        if (of_freq[1] == 0) of_freq[1] = 1;
    }

    if (n_seq < N / 64) {
        passes = 1;
    }

    struct OptNode {
        float cost = 1e30f;
        int lit_run = 0;
        int back_len = 1;
        int back_dist = 0;
        RepeatOffsets rep{};
    };
    constexpr float kInf = 1e30f;
    std::vector<OptNode> dp(static_cast<std::size_t>(N) + 1);

    std::vector<Seq> best_seqs;
    std::vector<std::uint8_t> best_literals;
    double best_est_bits = 1e30;

    if (passes < 1) passes = 1;
    for (int pass = 0; pass < passes; ++pass) {
        float lit_cost[256];
        bool used_huf = false;
        if (pass >= 1 && n_lit >= 18) {
            std::uint32_t u_lf[256];
            for (int s = 0; s < 256; ++s) u_lf[s] = static_cast<std::uint32_t>(lit_freq[s]);
            HufTable ht;
            if (huf_literals_table_from_freqs(u_lf, ht)) {
                const float unused_lit =
                    std::min(11.0f, static_cast<float>(ht.table_log + 1));
                for (int s = 0; s < 256; ++s) {
                    const int len = ht.lengths[static_cast<std::size_t>(s)];
                    lit_cost[s] = (len > 0) ? static_cast<float>(len) : unused_lit;
                }
                used_huf = true;
            }
        }
        if (!used_huf) {
            for (int s = 0; s < 256; ++s) {
                float c = sym_cost(lit_freq[s], n_lit, 11.0f);
                if (c < 1.0f) c = 1.0f;
                lit_cost[s] = c;
            }
        }

        float llc_cost[36];
        for (int c = 0; c < 36; ++c) {
            llc_cost[c] = sym_cost(ll_freq[c], n_seq, 9.5f) +
                          static_cast<float>(litlen_code_to_extra(c));
        }
        float mlc_cost[53];
        for (int c = 0; c < 53; ++c) {
            mlc_cost[c] = sym_cost(ml_freq[c], n_seq, 9.5f) +
                          static_cast<float>(matchlen_code_to_extra(c));
        }
        float ofc_cost[32];
        for (int c = 0; c < 32; ++c) {
            const int extra = (c == 0) ? 0 : (c == 1 ? 1 : c);
            ofc_cost[c] = sym_cost(of_freq[c], n_seq, 8.5f) + static_cast<float>(extra);
        }

        float ml_len_cost[65];
        for (int l = 3; l <= 64; ++l) {
            ml_len_cost[l] = mlc_cost[match_len_code(l).code];
        }

        auto ll_bits = [&](int ll) -> float {
            return llc_cost[lit_len_code(ll).code];
        };

        for (auto& node : dp) node.cost = kInf;
        dp[0].cost = ll_bits(0);
        dp[0].lit_run = 0;
        dp[0].back_len = 1;
        dp[0].back_dist = 0;
        dp[0].rep = initial_rep;

        struct RepCacheSlot {
            int dist = 0;
            int match_end = -1;
        };
        RepCacheSlot rep_cache[4];
        int rep_cache_next = 0;
        auto get_rep_len = [&](int i, std::size_t pos, int rd, int maxl) -> int {
            for (int c = 0; c < 4; ++c) {
                if (rep_cache[c].dist == rd && i < rep_cache[c].match_end) {
                    return std::min(maxl, rep_cache[c].match_end - i);
                }
            }
            const std::size_t cand = pos - static_cast<std::size_t>(rd);
            if (data[cand] != data[pos] || data[cand + 1] != data[pos + 1] ||
                data[cand + 2] != data[pos + 2]) {
                return 0;
            }
            const int rl = match_len_from(data + cand, data + pos, 3, maxl);
            rep_cache[rep_cache_next] = {rd, i + rl};
            rep_cache_next = (rep_cache_next + 1) & 3;
            return rl;
        };

        const float ll0_cost = ll_bits(0);
        for (int i = 0; i < N; ++i) {
            const OptNode cur = dp[static_cast<std::size_t>(i)];
            if (cur.cost >= kInf) continue;

            const std::size_t pos = start + static_cast<std::size_t>(i);
            const int ll = cur.lit_run;
            const float cur_ll_bits = ll_bits(ll);

            // 1. Literal transition (telescoping ll_bits delta).
            {
                const float nc = cur.cost + lit_cost[data[pos]] +
                                 (ll_bits(ll + 1) - cur_ll_bits);
                OptNode& next_node = dp[static_cast<std::size_t>(i + 1)];
                if (nc < next_node.cost) {
                    next_node.cost = nc;
                    next_node.lit_run = ll + 1;
                    next_node.rep = cur.rep;
                    next_node.back_len = 1;
                    next_node.back_dist = 0;
                }
            }

            const int maxl = std::min(kMaxMatch, N - i);
            if (maxl < kMinMatch) continue;

            const float base_seq_cost = cur.cost + ll0_cost;
            auto try_match = [&](int d, int max_match_len, float dc,
                                 const RepeatOffsets& next_rep, int min_l) {
                const float fixed_base = base_seq_cost + dc;
                const int dense_cap = (max_match_len >= sufficient_len) ? 16 : 48;
                const int dense_limit = std::min(max_match_len, dense_cap);
                for (int L = min_l; L <= dense_limit; ++L) {
                    const float mc = fixed_base + ml_len_cost[L];
                    OptNode& dst = dp[static_cast<std::size_t>(i + L)];
                    if (mc < dst.cost) {
                        dst.cost = mc;
                        dst.lit_run = 0;
                        dst.rep = next_rep;
                        dst.back_len = L;
                        dst.back_dist = d;
                    }
                }
                if (max_match_len > dense_limit) {
                    const int c_start = match_len_code(dense_limit + 1).code;
                    const int c_end = match_len_code(max_match_len).code;
                    for (int c = c_start; c <= c_end; ++c) {
                        const int L = std::min(ml_range_max[c], max_match_len);
                        if (L <= dense_limit) continue;
                        const float mc = fixed_base + mlc_cost[c];
                        OptNode& dst = dp[static_cast<std::size_t>(i + L)];
                        if (mc < dst.cost) {
                            dst.cost = mc;
                            dst.lit_run = 0;
                            dst.rep = next_rep;
                            dst.back_len = L;
                            dst.back_dist = d;
                        }
                    }
                }
            };

            // 2. Dynamic repeat-offset matches from `cur.rep`.
            {
                int rep_dists[3];
                if (ll > 0) {
                    rep_dists[0] = cur.rep.offsets[0];
                    rep_dists[1] = cur.rep.offsets[1];
                    rep_dists[2] = cur.rep.offsets[2];
                } else {
                    rep_dists[0] = cur.rep.offsets[1];
                    rep_dists[1] = cur.rep.offsets[2];
                    rep_dists[2] = (cur.rep.offsets[0] > 1)
                                       ? (cur.rep.offsets[0] - 1)
                                       : 0;
                }
                for (int r_idx = 0; r_idx < 3; ++r_idx) {
                    const int rd = rep_dists[r_idx];
                    if (rd <= 0 || rd > max_dist || static_cast<std::size_t>(rd) > pos) {
                        continue;
                    }
                    const int rl = get_rep_len(i, pos, rd, maxl);
                    if (rl >= kMinMatch) {
                        RepeatOffsets next_rep = cur.rep;
                        const Code oc = choose_offset(rd, ll, next_rep);
                        try_match(rd, rl, ofc_cost[oc.code], next_rep, kMinMatch);
                    }
                }
            }

            // 3. Pareto-optimal explicit matches from `pool`.
            {
                const int ci_begin = cand_start[static_cast<std::size_t>(i)];
                const int ci_end = cand_start[static_cast<std::size_t>(i) + 1];
                int prev_len = kMinMatch - 1;
                for (int ci = ci_begin; ci < ci_end; ++ci) {
                    const Match& c = pool[static_cast<std::size_t>(ci)];
                    const int max_l = std::min(c.length, maxl);
                    if (max_l <= prev_len) continue;
                    RepeatOffsets next_rep = cur.rep;
                    const Code oc = choose_offset(c.distance, ll, next_rep);
                    try_match(c.distance, max_l, ofc_cost[oc.code], next_rep,
                              prev_len + 1);
                    prev_len = max_l;
                }
            }
        }

        // Reconstruct sequences backward from N to 0.
        std::vector<Seq> seqs;
        int p = N;
        while (p > 0) {
            const OptNode& node = dp[static_cast<std::size_t>(p)];
            if (node.back_len <= 1) {
                --p;
            } else {
                const int match_start = p - node.back_len;
                const int lrun = dp[static_cast<std::size_t>(match_start)].lit_run;
                seqs.push_back(Seq{lrun, node.back_len, node.back_dist});
                p = match_start - lrun;
            }
        }
        std::reverse(seqs.begin(), seqs.end());

        std::vector<std::uint8_t> pass_lits;
        pass_lits.reserve(static_cast<std::size_t>(N));
        std::size_t cur_pos = start;
        for (const Seq& s : seqs) {
            if (s.lit_len > 0) {
                pass_lits.insert(pass_lits.end(), data + cur_pos,
                                 data + cur_pos + s.lit_len);
            }
            cur_pos += static_cast<std::size_t>(s.lit_len + s.match_len);
        }
        if (cur_pos < end) {
            pass_lits.insert(pass_lits.end(), data + cur_pos, data + end);
        }

        const double est_bits =
            estimate_block_total_bits(seqs, pass_lits, initial_rep);
        if (est_bits < best_est_bits || best_seqs.empty()) {
            best_est_bits = est_bits;
            best_seqs = std::move(seqs);
            best_literals = std::move(pass_lits);
        }

        if (pass + 1 < passes) {
            std::fill_n(lit_freq, 256, 0);
            std::fill_n(ll_freq, 36, 0);
            std::fill_n(of_freq, 32, 0);
            std::fill_n(ml_freq, 53, 0);
            const auto& eval_lits = (best_literals.empty() && N > 0) ? pass_lits : best_literals;
            const auto& eval_seqs = best_seqs;
            for (std::uint8_t b : eval_lits) lit_freq[b]++;
            n_lit = static_cast<int>(eval_lits.size());
            RepeatOffsets pass_rep = initial_rep;
            for (const Seq& s : eval_seqs) {
                ll_freq[lit_len_code(s.lit_len).code]++;
                const Code oc = choose_offset(s.match_dist, s.lit_len, pass_rep);
                of_freq[oc.code]++;
                ml_freq[match_len_code(s.match_len).code]++;
            }
            n_seq = static_cast<int>(eval_seqs.size());
        }
    }

    out_seqs = std::move(best_seqs);
    literals = std::move(best_literals);
    RepeatOffsets end_rep = initial_rep;
    for (const Seq& s : out_seqs) {
        choose_offset(s.match_dist, s.lit_len, end_rep);
    }
    rep = end_rep;
}

// --- Predefined normalized distributions (RFC 8878 §4.1.1) ---
constexpr std::int16_t kLLNorm[36] = {
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1};
constexpr std::int16_t kOFNorm[29] = {
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1};
constexpr std::int16_t kMLNorm[53] = {
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
    -1, -1, -1, -1, -1};

auto ll_ctable() -> const FseCTable& {
    static const FseCTable ct = build_fse_ctable(6, kLLNorm, 35);
    return ct;
}
auto of_ctable() -> const FseCTable& {
    static const FseCTable ct = build_fse_ctable(5, kOFNorm, 28);
    return ct;
}
auto ml_ctable() -> const FseCTable& {
    static const FseCTable ct = build_fse_ctable(6, kMLNorm, 52);
    return ct;
}

// --- Section emission ---
void put_literals_header(std::vector<std::byte>& out, int type, int size_format,
                         int regen, int csize, int k) {
    const std::uint64_t v = static_cast<std::uint64_t>(type) |
                            (static_cast<std::uint64_t>(size_format) << 2) |
                            (static_cast<std::uint64_t>(regen) << 4) |
                            (static_cast<std::uint64_t>(csize) << (4 + k));
    const int nbytes = (4 + 2 * k) / 8;
    for (int i = 0; i < nbytes; ++i) {
        out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
    }
}

auto literals_k(int regen, int csize) -> int {
    const int m = std::max(regen, csize);
    if (m <= 1023) return 10;
    if (m <= 16383) return 14;
    return 18;
}

// Emit the literals section using single-pass histogram + O(256) selection so
// only the winning table(s) are actually bit-encoded.
void emit_literals(std::vector<std::byte>& out,
                   const std::uint8_t* lits, int n, HufTable* prev_t) {
    if (n <= 0) {
        out.push_back(std::byte{0});
        return;
    }

    std::uint32_t freqs[256]{};
    int distinct = 0;
    for (int i = 0; i < n; ++i) {
        if (freqs[lits[i]] == 0) ++distinct;
        freqs[lits[i]]++;
    }

    // RLE literals.
    if (distinct == 1) {
        if (n <= 31) {
            out.push_back(static_cast<std::byte>((n << 3) | (0 << 2) | 1));
        } else if (n <= 4095) {
            out.push_back(static_cast<std::byte>(((n & 0xF) << 4) | (1 << 2) | 1));
            out.push_back(static_cast<std::byte>((n >> 4) & 0xFF));
        } else {
            out.push_back(static_cast<std::byte>(((n & 0xF) << 4) | (3 << 2) | 1));
            out.push_back(static_cast<std::byte>((n >> 4) & 0xFF));
            out.push_back(static_cast<std::byte>((n >> 12) & 0xFF));
        }
        out.push_back(static_cast<std::byte>(lits[0]));
        return;
    }

    // Try Huffman (Compressed / Treeless).
    if (n >= 18) {
        const bool single_stream = (n <= 1023);
        const std::size_t jump_bytes = single_stream ? 0 : 6;
        const std::size_t pad_bits = single_stream ? 1 : 4;

        auto estimate_stream_bytes = [&](const HufTable& tbl) -> std::size_t {
            if (tbl.table_log <= 0) return static_cast<std::size_t>(-1);
            std::size_t bits = 0;
            for (int s = 0; s < 256; ++s) {
                if (freqs[s] > 0) {
                    const int l = tbl.lengths[static_cast<std::size_t>(s)];
                    if (l == 0) return static_cast<std::size_t>(-1);
                    bits += static_cast<std::size_t>(freqs[s]) * static_cast<std::size_t>(l);
                }
            }
            return jump_bytes + ((bits + pad_bits + 7) >> 3);
        };

        HufTable new_t;
        std::vector<std::byte> new_weights;
        std::size_t fresh_est = static_cast<std::size_t>(n);
        if (huf_literals_table_from_freqs(freqs, new_t)) {
            const std::size_t sb = estimate_stream_bytes(new_t);
            if (sb < static_cast<std::size_t>(n) && huf_write_weights(new_t, new_weights)) {
                fresh_est = new_weights.size() + sb;
            }
        }

        std::size_t reuse_est = static_cast<std::size_t>(n);
        if (prev_t != nullptr && prev_t->table_log != 0) {
            const std::size_t sb = estimate_stream_bytes(*prev_t);
            if (sb < static_cast<std::size_t>(n)) reuse_est = sb;
        }

        const bool try_fresh = (fresh_est + 3 < static_cast<std::size_t>(n));
        const bool try_reuse = (reuse_est + 3 < static_cast<std::size_t>(n));

        if (try_fresh || try_reuse) {
            int best_type = 0;
            int best_sf = 0;
            int best_k = 0;
            std::vector<std::byte> best_body;
            std::size_t best_total = static_cast<std::size_t>(n);

            auto encode_cand = [&](int type, const HufTable& tbl,
                                   const std::vector<std::byte>* pre_weights) {
                std::vector<std::byte> body;
                if (type == 2) {
                    if (!huf_encode_streams(tbl, lits, n, body, single_stream)) return;
                    if (single_stream && pre_weights->size() + body.size() > 1023) return;
                    std::vector<std::byte> full;
                    full.reserve(pre_weights->size() + body.size());
                    full.insert(full.end(), pre_weights->begin(), pre_weights->end());
                    full.insert(full.end(), body.begin(), body.end());
                    body = std::move(full);
                } else {
                    if (!huf_encode_streams(tbl, lits, n, body, single_stream)) return;
                    if (single_stream && body.size() > 1023) return;
                }
                const int k = single_stream ? 10 : literals_k(n, static_cast<int>(body.size()));
                const int sf = single_stream ? 0 : ((k == 10) ? 1 : (k == 14) ? 2 : 3);
                const std::size_t hdr_bytes = static_cast<std::size_t>((4 + 2 * k) / 8);
                const std::size_t total = hdr_bytes + body.size();
                if (total < best_total) {
                    best_total = total;
                    best_type = type;
                    best_sf = sf;
                    best_k = k;
                    best_body = std::move(body);
                }
            };

            const bool close =
                try_fresh && try_reuse &&
                ((fresh_est > reuse_est ? fresh_est - reuse_est : reuse_est - fresh_est) <= 12);
            if (try_fresh && (!try_reuse || fresh_est <= reuse_est || close)) {
                encode_cand(2, new_t, &new_weights);
            }
            if (try_reuse && (!try_fresh || reuse_est < fresh_est || close)) {
                encode_cand(3, *prev_t, nullptr);
            }

            if (best_type != 0) {
                put_literals_header(out, best_type, best_sf, n,
                                    static_cast<int>(best_body.size()), best_k);
                out.insert(out.end(), best_body.begin(), best_body.end());
                if (best_type == 2 && prev_t != nullptr) *prev_t = std::move(new_t);
                return;
            }
        }
    }

    // Raw literals.
    if (n <= 31) {
        out.push_back(static_cast<std::byte>((n << 3)));
    } else if (n <= 4095) {
        out.push_back(static_cast<std::byte>(((n & 0xF) << 4) | (1 << 2)));
        out.push_back(static_cast<std::byte>((n >> 4) & 0xFF));
    } else {
        out.push_back(static_cast<std::byte>(((n & 0xF) << 4) | (3 << 2)));
        out.push_back(static_cast<std::byte>((n >> 4) & 0xFF));
        out.push_back(static_cast<std::byte>((n >> 12) & 0xFF));
    }
    const auto* bp = reinterpret_cast<const std::byte*>(lits);
    out.insert(out.end(), bp, bp + n);
}

// --------------------------------------------------------------------------
// Sequence FSE stream selection (O(alphabet) scoring + budgeted exact checks)
// --------------------------------------------------------------------------
struct SeqStream {
    int mode = 0;  // 0 predefined, 1 RLE, 2 FSE, 3 repeat
    int rle_sym = 0;
    FseCTable ct;
    std::vector<std::byte> desc;
};

auto fse_encoded_bits(const FseCTable& ct, const int* syms, int n) -> long long {
    if (n <= 0) return 0;
    if (n == 1) return ct.table_log;
    std::uint32_t value = fse_init_cstate2(ct, syms[n - 1]);
    long long bits = 0;
    for (int i = n - 2; i >= 0; --i) {
        const std::uint32_t d = ct.delta_nb_bits[static_cast<std::size_t>(syms[i])];
        const std::uint32_t nb = (value + d) >> 16;
        bits += nb;
        const int idx = static_cast<int>(value >> nb) +
                        ct.delta_find_state[static_cast<std::size_t>(syms[i])];
        value = ct.state_table[static_cast<std::size_t>(idx)];
    }
    bits += ct.table_log;
    return bits;
}

inline auto fse_ctable_supports(const FseCTable& ct, std::uint64_t used_mask,
                                const unsigned* counts, int max_present) -> bool {
    if (ct.table_log <= 0) return false;
    if (static_cast<std::size_t>(max_present) >= ct.delta_nb_bits.size()) {
        return false;
    }
    if (max_present < 64) {
        return (used_mask & ~ct.present_mask) == 0;
    }
    const auto max_nb = static_cast<std::uint32_t>(ct.table_log);
    for (int c = 0; c <= max_present; ++c) {
        if (counts[c] > 0 && (ct.delta_nb_bits[static_cast<std::size_t>(c)] >> 16) >= max_nb) {
            return false;
        }
    }
    return true;
}

auto fse_log2_lut() -> const double* {
    static const auto tbl = [] {
        std::array<double, 514> a{};
        a[0] = 0.0;
        for (int i = 1; i < 514; ++i) a[i] = std::log2(static_cast<double>(i));
        return a;
    }();
    return tbl.data();
}

auto make_seq_stream(const int* syms, int n, int max_code, int max_log,
                     const FseCTable& predef, const std::int16_t* predef_norm,
                     int predef_max, FseCTable* prev_ct, int level) -> SeqStream {
    SeqStream s;
    std::vector<unsigned> counts(static_cast<std::size_t>(max_code) + 1, 0);
    for (int i = 0; i < n; ++i) counts[static_cast<std::size_t>(syms[i])]++;

    int present = 0;
    int max_present = 0;
    std::uint64_t used_mask = 0;
    for (int c = 0; c <= max_code; ++c) {
        if (counts[static_cast<std::size_t>(c)] > 0) {
            ++present;
            max_present = c;
            if (c < 64) used_mask |= (1ULL << c);
        }
    }
    if (present <= 1) {
        s.mode = 1;  // RLE
        s.rle_sym = max_present;
        if (prev_ct != nullptr) prev_ct->table_log = 0;
        return s;
    }

    const double* l2 = fse_log2_lut();
    const bool predef_ok = fse_ctable_supports(predef, used_mask, counts.data(), max_present);
    const bool rep_ok =
        (prev_ct != nullptr && fse_ctable_supports(*prev_ct, used_mask, counts.data(), max_present));

    auto est_predef_bits = [&]() -> double {
        double bits = predef.table_log;
        for (int c = 0; c <= max_present && c <= predef_max; ++c) {
            if (counts[static_cast<std::size_t>(c)] > 0) {
                const int nv = predef_norm[c] == -1 ? 1 : predef_norm[c];
                bits += static_cast<double>(counts[static_cast<std::size_t>(c)]) *
                        (predef.table_log - l2[nv]);
            }
        }
        return bits;
    };

    if (level <= 5 && n < 16) {
        if (rep_ok) {
            s.mode = 3;
            s.ct = *prev_ct;
        } else {
            s.mode = 0;
            s.ct = predef;
        }
        if (prev_ct != nullptr) *prev_ct = s.ct;
        return s;
    }

    struct Mode2Cand {
        double est_bits = 1e30;
        int log = 0;
        std::vector<int> norm;
        std::vector<std::byte> desc;
    };
    std::vector<Mode2Cand> cands;

    if (n >= 16) {
        int min_log = 5;
        while ((1 << min_log) < present) ++min_log;
        if (min_log > max_log) min_log = max_log;

        auto score_norm = [&](const std::vector<int>& norm, int log) {
            int sum = 0;
            for (int c = 0; c <= max_present; ++c) {
                const int v = norm[static_cast<std::size_t>(c)];
                if (counts[static_cast<std::size_t>(c)] > 0 && v == 0) return;
                sum += (v == -1) ? 1 : v;
            }
            if (sum != (1 << log)) return;

            std::vector<std::byte> desc;
            fse_write_ncount(desc, norm.data(), max_present, log);
            double bits = static_cast<double>(desc.size()) * 8.0 + log;
            for (int c = 0; c <= max_present; ++c) {
                if (counts[static_cast<std::size_t>(c)] > 0) {
                    const int v = norm[static_cast<std::size_t>(c)];
                    const int slots = (v == -1) ? 1 : v;
                    bits += static_cast<double>(counts[static_cast<std::size_t>(c)]) *
                            (log - l2[slots]);
                }
            }
            cands.push_back(Mode2Cand{bits, log, norm, std::move(desc)});
        };

        for (int log = min_log; log <= max_log; ++log) {
            std::vector<int> norm;
            fse_normalize(counts.data(), n, max_present, log, norm);
            score_norm(norm, log);

            std::vector<int> norm_lp;
            fse_normalize_lowprob(counts.data(), n, max_present, log, norm_lp);
            if (norm_lp != norm) {
                score_norm(norm_lp, log);
            }
        }
        std::sort(cands.begin(), cands.end(), [](const Mode2Cand& a, const Mode2Cand& b) {
            return a.est_bits < b.est_bits;
        });
    }

    // Fast path (level <= 5): pick single best O(alphabet) winner directly.
    if (level <= 5) {
        double best_est = 1e30;
        if (predef_ok) {
            best_est = est_predef_bits();
            s.mode = 0;
            s.ct = predef;
        }
        if (rep_ok) {
            // Evaluate exact repeat bits only when repeat is plausible (no norm array stored).
            const long long rep_bits = fse_encoded_bits(*prev_ct, syms, n);
            if (static_cast<double>(rep_bits) < best_est) {
                best_est = static_cast<double>(rep_bits);
                s.mode = 3;
                s.ct = *prev_ct;
            }
        }
        if (!cands.empty() && cands[0].est_bits + 8.0 < best_est) {
            const auto& win = cands[0];
            std::vector<std::int16_t> norm16(static_cast<std::size_t>(max_present) + 1);
            for (int c = 0; c <= max_present; ++c) {
                norm16[static_cast<std::size_t>(c)] =
                    static_cast<std::int16_t>(win.norm[static_cast<std::size_t>(c)]);
            }
            s.mode = 2;
            s.ct = build_fse_ctable(win.log, norm16.data(), max_present);
            s.desc = std::move(cands[0].desc);
        }
        if (prev_ct != nullptr) *prev_ct = s.ct;
        return s;
    }

    long long best_bits = (1LL << 62);
    if (predef_ok) {
        best_bits = fse_encoded_bits(predef, syms, n);
        s.mode = 0;
        s.ct = predef;
    }
    if (rep_ok) {
        const long long rep_bits = fse_encoded_bits(*prev_ct, syms, n);
        if (rep_bits < best_bits) {
            best_bits = rep_bits;
            s.mode = 3;
            s.ct = *prev_ct;
        }
    }

    const int top_k = (level >= 16) ? 3 : 2;
    const int eval_k = std::min<int>(top_k, static_cast<int>(cands.size()));
    std::vector<std::int16_t> norm16(static_cast<std::size_t>(max_present) + 1);
    for (int i = 0; i < eval_k; ++i) {
        auto& cand = cands[static_cast<std::size_t>(i)];
        for (int c = 0; c <= max_present; ++c) {
            norm16[static_cast<std::size_t>(c)] =
                static_cast<std::int16_t>(cand.norm[static_cast<std::size_t>(c)]);
        }
        FseCTable ct = build_fse_ctable(cand.log, norm16.data(), max_present);
        const long long bits =
            static_cast<long long>(cand.desc.size()) * 8 +
            fse_encoded_bits(ct, syms, n);
        if (bits < best_bits) {
            best_bits = bits;
            s.mode = 2;
            s.ct = std::move(ct);
            s.desc = std::move(cand.desc);
        }
    }

    if (prev_ct != nullptr) *prev_ct = s.ct;
    return s;
}

void emit_seq_table(std::vector<std::byte>& out, const SeqStream& s) {
    if (s.mode == 1) {
        out.push_back(static_cast<std::byte>(s.rle_sym));
    } else if (s.mode == 2) {
        out.insert(out.end(), s.desc.begin(), s.desc.end());
    }
}

void emit_sequences(std::vector<std::byte>& out, std::span<const Seq> seqs,
                    RepeatOffsets& rep, FseCTable* prev_ll, FseCTable* prev_of,
                    FseCTable* prev_ml, int level) {
    const int n = static_cast<int>(seqs.size());
    if (n == 0) {
        out.push_back(std::byte{0});
        return;
    }
    if (n < 128) {
        out.push_back(static_cast<std::byte>(n));
    } else if (n <= 0x7EFF) {
        out.push_back(static_cast<std::byte>(128 + (n >> 8)));
        out.push_back(static_cast<std::byte>(n & 0xFF));
    } else {
        const int v = n - 0x7F00;
        out.push_back(static_cast<std::byte>(255));
        out.push_back(static_cast<std::byte>(v & 0xFF));
        out.push_back(static_cast<std::byte>((v >> 8) & 0xFF));
    }

    std::vector<int> llc(static_cast<std::size_t>(n));
    std::vector<int> ofc(static_cast<std::size_t>(n));
    std::vector<int> mlc(static_cast<std::size_t>(n));
    std::vector<Code> ofcodes(static_cast<std::size_t>(n));
    RepeatOffsets rep_next = rep;
    for (int i = 0; i < n; ++i) {
        const Seq& s = seqs[static_cast<std::size_t>(i)];
        llc[static_cast<std::size_t>(i)] = lit_len_code(s.lit_len).code;
        ofcodes[static_cast<std::size_t>(i)] =
            choose_offset(s.match_dist, s.lit_len, rep_next);
        ofc[static_cast<std::size_t>(i)] = ofcodes[static_cast<std::size_t>(i)].code;
        mlc[static_cast<std::size_t>(i)] = match_len_code(s.match_len).code;
    }
    rep = rep_next;

    SeqStream ll = make_seq_stream(llc.data(), n, 35, 9, ll_ctable(), kLLNorm, 35, prev_ll, level);
    SeqStream of = make_seq_stream(ofc.data(), n, 31, 8, of_ctable(), kOFNorm, 28, prev_of, level);
    SeqStream ml = make_seq_stream(mlc.data(), n, 52, 9, ml_ctable(), kMLNorm, 52, prev_ml, level);

    out.push_back(static_cast<std::byte>((ll.mode << 6) | (of.mode << 4) | (ml.mode << 2)));
    emit_seq_table(out, ll);
    emit_seq_table(out, of);
    emit_seq_table(out, ml);

    FseBitWriter w;
    w.reserve(static_cast<std::size_t>(n) * 3 + 32);
    std::uint32_t ll_state = 0, of_state = 0, ml_state = 0;

    {
        const Seq& s = seqs[static_cast<std::size_t>(n - 1)];
        const Code lc = lit_len_code(s.lit_len);
        const Code mc = match_len_code(s.match_len);
        const Code oc = ofcodes[static_cast<std::size_t>(n - 1)];
        if (ll.mode != 1) ll_state = fse_init_cstate2(ll.ct, lc.code);
        if (of.mode != 1) of_state = fse_init_cstate2(of.ct, oc.code);
        if (ml.mode != 1) ml_state = fse_init_cstate2(ml.ct, mc.code);
        if (lc.extra_bits) w.put_bits(static_cast<std::uint32_t>(lc.extra_val), lc.extra_bits);
        if (mc.extra_bits) w.put_bits(static_cast<std::uint32_t>(mc.extra_val), mc.extra_bits);
        if (oc.extra_bits) w.put_bits(static_cast<std::uint32_t>(oc.extra_val), oc.extra_bits);
    }
    for (int i = n - 2; i >= 0; --i) {
        const Seq& s = seqs[static_cast<std::size_t>(i)];
        const Code lc = lit_len_code(s.lit_len);
        const Code mc = match_len_code(s.match_len);
        const Code oc = ofcodes[static_cast<std::size_t>(i)];
        if (of.mode != 1) fse_encode_symbol(w, of.ct, of_state, oc.code);
        if (ml.mode != 1) fse_encode_symbol(w, ml.ct, ml_state, mc.code);
        if (ll.mode != 1) fse_encode_symbol(w, ll.ct, ll_state, lc.code);
        if (lc.extra_bits) w.put_bits(static_cast<std::uint32_t>(lc.extra_val), lc.extra_bits);
        if (mc.extra_bits) w.put_bits(static_cast<std::uint32_t>(mc.extra_val), mc.extra_bits);
        if (oc.extra_bits) w.put_bits(static_cast<std::uint32_t>(oc.extra_val), oc.extra_bits);
    }
    if (ml.mode != 1) fse_flush_cstate(w, ml.ct, ml_state);
    if (of.mode != 1) fse_flush_cstate(w, of.ct, of_state);
    if (ll.mode != 1) fse_flush_cstate(w, ll.ct, ll_state);
    w.put_bit(true);
    w.align_to_byte();
    const auto& bytes = w.data();
    out.insert(out.end(), bytes.begin(), bytes.end());
}

// --------------------------------------------------------------------------
// Adaptive MDL block splitting for Zstd (levels >= 9)
// --------------------------------------------------------------------------
auto estimate_zstd_subblock_bytes(const std::uint32_t lit_freq[256], int n_lit,
                                  const std::uint32_t ll_freq[36],
                                  const std::uint32_t of_freq[32],
                                  const std::uint32_t ml_freq[53], int n_seq,
                                  double extra_bits, std::size_t raw_len) -> double {
    double lit_bytes = static_cast<double>(n_lit) + 3.0;
    if (n_lit >= 18) {
        int distinct = 0;
        for (int s = 0; s < 256; ++s) {
            if (lit_freq[s] > 0) ++distinct;
        }
        if (distinct == 1) {
            lit_bytes = 3.0;
        } else if (distinct >= 2) {
            HufTable ht;
            if (huf_literals_table_from_freqs(lit_freq, ht)) {
                double hb = static_cast<double>(distinct) * 0.55 + 4.0;
                for (int s = 0; s < 256; ++s) {
                    if (lit_freq[s] > 0) {
                        hb += static_cast<double>(lit_freq[s]) *
                              static_cast<double>(ht.lengths[static_cast<std::size_t>(s)]) * 0.125;
                    }
                }
                if (hb < lit_bytes) lit_bytes = hb;
            }
        }
    }

    if (n_seq <= 0) {
        return std::min(static_cast<double>(raw_len), lit_bytes + 1.0) + 3.0;
    }

    auto stream_bits = [n_seq](const std::uint32_t* freq, int max_c) -> double {
        int present = 0;
        for (int c = 0; c <= max_c; ++c) {
            if (freq[c] > 0) ++present;
        }
        if (present <= 1) return 8.0;
        double bits = 0.0;
        const double log_tot = std::log2(static_cast<double>(n_seq));
        for (int c = 0; c <= max_c; ++c) {
            if (freq[c] > 0) {
                bits += static_cast<double>(freq[c]) *
                        (log_tot - std::log2(static_cast<double>(freq[c])));
            }
        }
        const double hdr = (n_seq >= 16) ? std::min(160.0, static_cast<double>(present) * 5.0 + 12.0) : 0.0;
        return bits + hdr;
    };

    const double seq_bytes =
        3.0 + (extra_bits + stream_bits(ll_freq, 35) + stream_bits(of_freq, 31) +
               stream_bits(ml_freq, 52)) * 0.125;
    const double comp_bytes = lit_bytes + seq_bytes;
    return std::min(static_cast<double>(raw_len), comp_bytes) + 3.0;
}

struct ZstdSplitRange {
    std::size_t seq_a = 0;
    std::size_t seq_b = 0;
    std::size_t lit_a = 0;
    std::size_t lit_b = 0;
    std::size_t raw_a = 0;
    std::size_t raw_b = 0;
};

void find_zstd_splits_rec(const std::vector<Seq>& seqs,
                          const std::vector<std::uint8_t>& lits,
                          const std::vector<std::size_t>& seq_lit_off,
                          const std::vector<std::size_t>& seq_raw_off,
                          const std::vector<int>& seq_of_code,
                          const std::vector<int>& seq_extra_bits,
                          std::size_t sa, std::size_t sb,
                          std::size_t total_lits_end, std::size_t total_raw_end,
                          std::vector<std::size_t>& splits) {
    constexpr std::size_t kMinSplitSeqs = 128;
    constexpr int kChunks = 16;
    const std::size_t n = sb - sa;
    if (n < kMinSplitSeqs * 2) return;

    struct ChunkStats {
        std::uint32_t lit_freq[256]{};
        std::uint32_t ll_freq[36]{};
        std::uint32_t of_freq[32]{};
        std::uint32_t ml_freq[53]{};
        double extra_bits = 0.0;
        int n_lit = 0;
    };

    std::vector<ChunkStats> pref(kChunks + 1);
    for (int k = 0; k < kChunks; ++k) {
        pref[static_cast<std::size_t>(k + 1)] = pref[static_cast<std::size_t>(k)];
        auto& cs = pref[static_cast<std::size_t>(k + 1)];
        const std::size_t s0 = sa + (n * static_cast<std::size_t>(k)) / kChunks;
        const std::size_t s1 = sa + (n * static_cast<std::size_t>(k + 1)) / kChunks;
        for (std::size_t i = s0; i < s1; ++i) {
            const Seq& s = seqs[i];
            cs.ll_freq[lit_len_code(s.lit_len).code]++;
            cs.of_freq[seq_of_code[i]]++;
            cs.ml_freq[match_len_code(s.match_len).code]++;
            cs.extra_bits += static_cast<double>(seq_extra_bits[i]);
        }
        const std::size_t l0 = seq_lit_off[s0];
        const std::size_t l1 = (s1 == sb) ? total_lits_end : seq_lit_off[s1];
        for (std::size_t li = l0; li < l1; ++li) {
            cs.lit_freq[lits[li]]++;
        }
        cs.n_lit += static_cast<int>(l1 - l0);
    }

    const std::size_t raw_start = seq_raw_off[sa];
    const auto& whole = pref[kChunks];
    const double whole_bytes = estimate_zstd_subblock_bytes(
        whole.lit_freq, whole.n_lit, whole.ll_freq, whole.of_freq,
        whole.ml_freq, static_cast<int>(n), whole.extra_bits,
        total_raw_end - raw_start);

    double best_bytes = whole_bytes;
    int best_k = -1;
    for (int k = 1; k < kChunks; ++k) {
        const std::size_t split = sa + (n * static_cast<std::size_t>(k)) / kChunks;
        if (split - sa < kMinSplitSeqs || sb - split < kMinSplitSeqs) continue;
        const auto& left = pref[static_cast<std::size_t>(k)];
        ChunkStats right;
        for (int s = 0; s < 256; ++s) right.lit_freq[s] = whole.lit_freq[s] - left.lit_freq[s];
        for (int s = 0; s < 36; ++s) right.ll_freq[s] = whole.ll_freq[s] - left.ll_freq[s];
        for (int s = 0; s < 32; ++s) right.of_freq[s] = whole.of_freq[s] - left.of_freq[s];
        for (int s = 0; s < 53; ++s) right.ml_freq[s] = whole.ml_freq[s] - left.ml_freq[s];
        right.extra_bits = whole.extra_bits - left.extra_bits;
        right.n_lit = whole.n_lit - left.n_lit;

        const std::size_t mid_raw = seq_raw_off[split];
        const double cand_bytes =
            estimate_zstd_subblock_bytes(
                left.lit_freq, left.n_lit, left.ll_freq, left.of_freq,
                left.ml_freq, static_cast<int>(split - sa), left.extra_bits,
                mid_raw - raw_start) +
            estimate_zstd_subblock_bytes(
                right.lit_freq, right.n_lit, right.ll_freq, right.of_freq,
                right.ml_freq, static_cast<int>(sb - split), right.extra_bits,
                total_raw_end - mid_raw);
        if (cand_bytes + 24.0 < best_bytes) {
            best_bytes = cand_bytes;
            best_k = k;
        }
    }

    if (best_k <= 0) return;

    std::size_t mid = sa + (n * static_cast<std::size_t>(best_k)) / kChunks;
    ChunkStats mid_left = pref[static_cast<std::size_t>(best_k)];

    // 2-step local refinement around `mid`.
    std::size_t step = std::max<std::size_t>(32, n / 32);
    for (int iter = 0; iter < 2 && step >= 16; ++iter, step >>= 1) {
        std::size_t best_cand = mid;
        ChunkStats best_left = mid_left;
        double best_pair_bytes = best_bytes;
        for (int dir : {-1, 1}) {
            if (dir < 0 && mid < sa + kMinSplitSeqs + step) continue;
            if (dir > 0 && mid + step + kMinSplitSeqs > sb) continue;
            const std::size_t cand = (dir < 0) ? (mid - step) : (mid + step);
            ChunkStats cl = mid_left;
            if (cand > mid) {
                for (std::size_t i = mid; i < cand; ++i) {
                    const Seq& s = seqs[i];
                    cl.ll_freq[lit_len_code(s.lit_len).code]++;
                    cl.of_freq[seq_of_code[i]]++;
                    cl.ml_freq[match_len_code(s.match_len).code]++;
                    cl.extra_bits += static_cast<double>(seq_extra_bits[i]);
                }
                for (std::size_t li = seq_lit_off[mid]; li < seq_lit_off[cand]; ++li) {
                    cl.lit_freq[lits[li]]++;
                    ++cl.n_lit;
                }
            } else {
                for (std::size_t i = cand; i < mid; ++i) {
                    const Seq& s = seqs[i];
                    cl.ll_freq[lit_len_code(s.lit_len).code]--;
                    cl.of_freq[seq_of_code[i]]--;
                    cl.ml_freq[match_len_code(s.match_len).code]--;
                    cl.extra_bits -= static_cast<double>(seq_extra_bits[i]);
                }
                for (std::size_t li = seq_lit_off[cand]; li < seq_lit_off[mid]; ++li) {
                    cl.lit_freq[lits[li]]--;
                    --cl.n_lit;
                }
            }
            ChunkStats cr;
            for (int s = 0; s < 256; ++s) cr.lit_freq[s] = whole.lit_freq[s] - cl.lit_freq[s];
            for (int s = 0; s < 36; ++s) cr.ll_freq[s] = whole.ll_freq[s] - cl.ll_freq[s];
            for (int s = 0; s < 32; ++s) cr.of_freq[s] = whole.of_freq[s] - cl.of_freq[s];
            for (int s = 0; s < 53; ++s) cr.ml_freq[s] = whole.ml_freq[s] - cl.ml_freq[s];
            cr.extra_bits = whole.extra_bits - cl.extra_bits;
            cr.n_lit = whole.n_lit - cl.n_lit;

            const std::size_t cr_raw = seq_raw_off[cand];
            const double cb =
                estimate_zstd_subblock_bytes(
                    cl.lit_freq, cl.n_lit, cl.ll_freq, cl.of_freq,
                    cl.ml_freq, static_cast<int>(cand - sa), cl.extra_bits,
                    cr_raw - raw_start) +
                estimate_zstd_subblock_bytes(
                    cr.lit_freq, cr.n_lit, cr.ll_freq, cr.of_freq,
                    cr.ml_freq, static_cast<int>(sb - cand), cr.extra_bits,
                    total_raw_end - cr_raw);
            if (cb < best_pair_bytes) {
                best_pair_bytes = cb;
                best_cand = cand;
                best_left = cl;
            }
        }
        if (best_cand != mid) {
            mid = best_cand;
            mid_left = best_left;
            best_bytes = best_pair_bytes;
        }
    }

    const std::size_t mid_lit = seq_lit_off[mid];
    const std::size_t mid_raw = seq_raw_off[mid];
    find_zstd_splits_rec(seqs, lits, seq_lit_off, seq_raw_off, seq_of_code,
                         seq_extra_bits, sa, mid, mid_lit, mid_raw, splits);
    splits.push_back(mid);
    find_zstd_splits_rec(seqs, lits, seq_lit_off, seq_raw_off, seq_of_code,
                         seq_extra_bits, mid, sb, total_lits_end, total_raw_end,
                         splits);
}

auto compute_subblocks(const std::vector<Seq>& seqs,
                       const std::vector<std::uint8_t>& lits,
                       std::size_t block_len,
                       const RepeatOffsets& start_rep) -> std::vector<ZstdSplitRange> {
    const std::size_t n = seqs.size();
    if (n < 256) {
        return {ZstdSplitRange{0, n, 0, lits.size(), 0, block_len}};
    }

    std::vector<std::size_t> seq_lit_off(n + 1, 0);
    std::vector<std::size_t> seq_raw_off(n + 1, 0);
    std::vector<int> seq_of_code(n, 0);
    std::vector<int> seq_extra_bits(n, 0);
    RepeatOffsets r = start_rep;
    std::size_t loff = 0;
    std::size_t roff = 0;
    for (std::size_t i = 0; i < n; ++i) {
        seq_lit_off[i] = loff;
        seq_raw_off[i] = roff;
        const Seq& s = seqs[i];
        const Code lc = lit_len_code(s.lit_len);
        const Code oc = choose_offset(s.match_dist, s.lit_len, r);
        const Code mc = match_len_code(s.match_len);
        seq_of_code[i] = oc.code;
        seq_extra_bits[i] = lc.extra_bits + oc.extra_bits + mc.extra_bits;
        loff += static_cast<std::size_t>(s.lit_len);
        roff += static_cast<std::size_t>(s.lit_len + s.match_len);
    }
    seq_lit_off[n] = lits.size();
    seq_raw_off[n] = block_len;

    std::vector<std::size_t> splits;
    splits.push_back(0);
    find_zstd_splits_rec(seqs, lits, seq_lit_off, seq_raw_off, seq_of_code,
                         seq_extra_bits, 0, n, lits.size(), block_len, splits);
    splits.push_back(n);

    std::vector<ZstdSplitRange> ranges;
    ranges.reserve(splits.size() - 1);
    for (std::size_t k = 0; k + 1 < splits.size(); ++k) {
        const std::size_t sa = splits[k];
        const std::size_t sb = splits[k + 1];
        const std::size_t la = seq_lit_off[sa];
        const std::size_t lb = (sb == n) ? lits.size() : seq_lit_off[sb];
        const std::size_t ra = seq_raw_off[sa];
        const std::size_t rb = (sb == n) ? block_len : seq_raw_off[sb];
        ranges.push_back(ZstdSplitRange{sa, sb, la, lb, ra, rb});
    }
    return ranges;
}

}  // namespace

auto compress(std::span<const std::byte> data, int level,
              int /*long_distance_log*/) -> std::vector<std::byte> {
    if (data.empty()) return {};
    if (level < 1) level = 1;
    if (level > 22) level = 22;

    const std::size_t size = data.size();

    // Window log per level tier (continuous cross-block window for all levels).
    int window_log = 19;
    if (level <= 4) window_log = 19;
    else if (level <= 9) window_log = 20;
    else if (level <= 18) window_log = 21;
    else if (level <= 20) window_log = 22;
    else window_log = (size > (8ULL << 20)) ? 24 : 23;

    auto next_pow2 = [](std::size_t v) -> std::size_t {
        if (v <= 1024) return 1024;
        return std::size_t{1} << std::bit_width(v - 1);
    };
    const std::size_t max_win = std::size_t{1} << window_log;
    const int actual_window = static_cast<int>(
        std::min(max_win, next_pow2(std::max<std::size_t>(size, 1024))));

    std::vector<std::byte> output;
    output.reserve(size / 2 + 64);

    const std::uint32_t magic = 0xFD2FB528u;
    const std::byte magic_bytes[4] = {
        static_cast<std::byte>(magic & 0xFFu),
        static_cast<std::byte>((magic >> 8) & 0xFFu),
        static_cast<std::byte>((magic >> 16) & 0xFFu),
        static_cast<std::byte>((magic >> 24) & 0xFFu),
    };
    output.insert(output.end(), magic_bytes, magic_bytes + 4);

    int fcs_size;
    std::uint8_t fcs_code;
    std::uint64_t fcs_value;
    if (size < 256) {
        fcs_code = 0; fcs_size = 1; fcs_value = size;
    } else if (size < 256 + 65536) {
        fcs_code = 1; fcs_size = 2; fcs_value = size - 256;
    } else if (size < (1ULL << 32)) {
        fcs_code = 2; fcs_size = 4; fcs_value = size;
    } else {
        fcs_code = 3; fcs_size = 8; fcs_value = size;
    }

    const bool single_seg = (fcs_code == 0) || size <= max_win;
    const std::uint8_t desc = static_cast<std::uint8_t>(
        (single_seg ? (1 << 5) : 0) | (fcs_code << 6));
    output.push_back(static_cast<std::byte>(desc));
    if (!single_seg) {
        output.push_back(static_cast<std::byte>((window_log - 10) << 3));
    }
    for (int i = 0; i < fcs_size; ++i) {
        output.push_back(static_cast<std::byte>((fcs_value >> (8 * i)) & 0xFF));
    }

    const auto* p = reinterpret_cast<const std::uint8_t*>(data.data());
    const bool is_tier1 = (level <= 2);
    const bool is_tier2 = (level >= 3 && level <= 5);
    const bool is_tier3 = (level >= 6 && level <= 15);
    const bool is_tier4 = (level >= 16);

    int effort = 16;
    int nice_len = 64;
    int lazy_depth = 1;
    int bt_depth = 48;
    int passes = 2;
    if (is_tier1) {
        effort = (level == 1) ? 2 : 3;
    } else if (is_tier2) {
        effort = (level == 3) ? 4 : (level == 4 ? 8 : 12);
        nice_len = (level == 3) ? 32 : (level == 4 ? 48 : 64);
    } else if (is_tier3) {
        switch (level) {
            case 6:  effort = 24;  nice_len = 64;  lazy_depth = 1; break;
            case 7:  effort = 48;  nice_len = 96;  lazy_depth = 1; break;
            case 8:  effort = 64;  nice_len = 128; lazy_depth = 1; break;
            case 9:  effort = 96;  nice_len = 192; lazy_depth = 1; break;
            case 10: effort = 128; nice_len = 256; lazy_depth = 2; break;
            case 11: effort = 192; nice_len = 256; lazy_depth = 2; break;
            case 12: effort = 256; nice_len = 256; lazy_depth = 2; break;
            case 13: effort = 320; nice_len = 256; lazy_depth = 2; break;
            case 14: effort = 384; nice_len = 256; lazy_depth = 2; break;
            default: effort = 512; nice_len = 256; lazy_depth = 2; break;
        }
    } else if (is_tier4) {
        switch (level) {
            case 16: bt_depth = 32;  nice_len = 256; passes = 2; break;
            case 17: bt_depth = 48;  nice_len = 256; passes = 2; break;
            case 18: bt_depth = 48;  nice_len = 256; passes = 2; break;
            case 19: bt_depth = 64;  nice_len = 256; passes = 3; break;
            case 20: bt_depth = 128; nice_len = 512; passes = 3; break;
            case 21: bt_depth = 128; nice_len = 512; passes = 4; break;
            default: bt_depth = 256; nice_len = 512; passes = 4; break;
        }
    }

    ZstdWorkspace& ws = get_zstd_workspace();
    const std::int32_t base = ws.begin_frame(
        size, actual_window,
        /*need_h3=*/true,
        /*need_h4_chain=*/(is_tier1 || is_tier2 || is_tier3),
        /*need_h6=*/(is_tier2 || is_tier3),
        /*need_bt=*/is_tier4);

    RepeatOffsets match_rep;
    RepeatOffsets emit_rep;
    HufTable prev_huf;
    FseCTable prev_ll, prev_of, prev_ml;

    std::vector<Seq> seqs;
    std::vector<std::uint8_t> literals;
    std::vector<std::byte> content;

    auto emit_block_header = [&](bool last, std::uint32_t type, std::uint32_t cs) {
        const std::uint32_t blk_hdr = (last ? 1u : 0u) | (type << 1) | (cs << 3);
        output.push_back(static_cast<std::byte>(blk_hdr & 0xFF));
        output.push_back(static_cast<std::byte>((blk_hdr >> 8) & 0xFF));
        output.push_back(static_cast<std::byte>((blk_hdr >> 16) & 0xFF));
    };

    const std::size_t mask = static_cast<std::size_t>(actual_window - 1);

    for (std::size_t off = 0; off < size;) {
        const std::size_t blen = std::min<std::size_t>(kBlockSize, size - off);
        const std::size_t bend = off + blen;
        const bool is_last = (bend == size);

        if (is_uniform_bytes(p + off, blen)) {
            emit_block_header(is_last, 1u, static_cast<std::uint32_t>(blen));
            output.push_back(static_cast<std::byte>(p[off]));
            // Seed a small tail into match tables if more blocks follow.
            if (!is_last) {
                const std::size_t seed_from =
                    (blen > 256) ? (bend - 256) : off;
                for (std::size_t sp = seed_from; sp < bend; ++sp) {
                    if (is_tier1 && sp + 3 <= bend) {
                        const std::uint32_t h3 = hash3(p + sp);
                        ws.prev3[sp & mask] = ws.head3[h3];
                        ws.head3[h3] = base + static_cast<std::int32_t>(sp);
                        if (sp + 4 <= bend) {
                            const std::uint32_t h4 = hash4(p + sp, 17);
                            ws.prev4[sp & mask] = ws.head4[h4];
                            ws.head4[h4] = base + static_cast<std::int32_t>(sp);
                        }
                    } else if (is_tier2 && sp + 3 <= bend) {
                        ws.head3[hash3(p + sp)] = base + static_cast<std::int32_t>(sp);
                        if (sp + 4 <= bend) {
                            const std::uint32_t h4 = hash4(p + sp, 17);
                            ws.prev4[sp & mask] = ws.head4[h4];
                            ws.head4[h4] = base + static_cast<std::int32_t>(sp);
                            if (sp + 6 <= bend) {
                                ws.head6[hash6(p + sp, 17)] = base + static_cast<std::int32_t>(sp);
                            }
                        }
                    } else if (is_tier3 && sp + 3 <= bend) {
                        const std::uint32_t h3 = hash3(p + sp);
                        ws.prev3[sp & mask] = ws.head3[h3];
                        ws.head3[h3] = base + static_cast<std::int32_t>(sp);
                        if (sp + 4 <= bend) {
                            const std::uint32_t h4 = hash4(p + sp, 17);
                            ws.prev4[sp & mask] = ws.head4[h4];
                            ws.head4[h4] = base + static_cast<std::int32_t>(sp);
                            if (sp + 6 <= bend) {
                                ws.head6[hash6(p + sp, 17)] = base + static_cast<std::int32_t>(sp);
                            }
                        }
                    } else if (is_tier4) {
                        const std::size_t min_p =
                            (sp > static_cast<std::size_t>(actual_window - 1))
                                ? (sp - static_cast<std::size_t>(actual_window - 1))
                                : 0;
                        bt_skip_insert(p, bend, sp, ws, base, mask, min_p, 8, 64);
                    }
                }
            }
            off = bend;
            continue;
        }

        const RepeatOffsets block_start_rep = match_rep;
        if (is_tier1) {
            match_tier1_block(p, off, bend, ws, base, actual_window, effort,
                              seqs, literals, match_rep);
        } else if (is_tier2) {
            match_tier2_block(p, off, bend, ws, base, actual_window, effort,
                              nice_len, seqs, literals, match_rep);
        } else if (is_tier3) {
            match_tier3_block(p, off, bend, ws, base, actual_window, effort,
                              nice_len, lazy_depth, seqs, literals, match_rep);
        } else {
            build_sequences_optimal(p, off, bend, ws, base, actual_window,
                                    bt_depth, nice_len, passes, seqs, literals,
                                    match_rep);
        }

        // Seed the last 3 boundary positions across `size` for subsequent blocks.
        if (!is_last) {
            const std::size_t tail_from = (bend >= 3) ? (bend - 3) : off;
            for (std::size_t sp = tail_from; sp < bend; ++sp) {
                if (is_tier1 && sp + 3 <= size) {
                    const std::uint32_t h3 = hash3(p + sp);
                    ws.prev3[sp & mask] = ws.head3[h3];
                    ws.head3[h3] = base + static_cast<std::int32_t>(sp);
                    if (sp + 4 <= size) {
                        const std::uint32_t h4 = hash4(p + sp, 17);
                        ws.prev4[sp & mask] = ws.head4[h4];
                        ws.head4[h4] = base + static_cast<std::int32_t>(sp);
                    }
                } else if (is_tier2 && sp + 3 <= size) {
                    ws.head3[hash3(p + sp)] = base + static_cast<std::int32_t>(sp);
                    if (sp + 4 <= size) {
                        const std::uint32_t h4 = hash4(p + sp, 17);
                        ws.prev4[sp & mask] = ws.head4[h4];
                        ws.head4[h4] = base + static_cast<std::int32_t>(sp);
                        if (sp + 6 <= size) {
                            ws.head6[hash6(p + sp, 17)] = base + static_cast<std::int32_t>(sp);
                        }
                    }
                } else if (is_tier3 && sp + 3 <= size) {
                    const std::uint32_t h3 = hash3(p + sp);
                    ws.prev3[sp & mask] = ws.head3[h3];
                    ws.head3[h3] = base + static_cast<std::int32_t>(sp);
                    if (sp + 4 <= size) {
                        const std::uint32_t h4 = hash4(p + sp, 17);
                        ws.prev4[sp & mask] = ws.head4[h4];
                        ws.head4[h4] = base + static_cast<std::int32_t>(sp);
                        if (sp + 6 <= size) {
                            ws.head6[hash6(p + sp, 17)] = base + static_cast<std::int32_t>(sp);
                        }
                    }
                } else if (is_tier4) {
                    const std::size_t min_p =
                        (sp > static_cast<std::size_t>(actual_window - 1))
                            ? (sp - static_cast<std::size_t>(actual_window - 1))
                            : 0;
                    bt_skip_insert(p, size, sp, ws, base, mask, min_p, 8, 64);
                }
            }
        }

        std::vector<ZstdSplitRange> ranges;
        if (level >= 9 && seqs.size() >= 256) {
            ranges = compute_subblocks(seqs, literals, blen, block_start_rep);
        } else {
            ranges.push_back(ZstdSplitRange{0, seqs.size(), 0, literals.size(), 0, blen});
        }

        for (std::size_t ri = 0; ri < ranges.size(); ++ri) {
            const auto& rg = ranges[ri];
            const bool sub_last = is_last && (ri + 1 == ranges.size());
            const std::size_t sub_raw_len = rg.raw_b - rg.raw_a;
            const std::uint8_t* sub_raw = p + off + rg.raw_a;
            const std::span<const Seq> sub_seqs{
                seqs.data() + rg.seq_a, rg.seq_b - rg.seq_a};

            bool emitted = false;
            if (!sub_seqs.empty()) {
                content.clear();
                HufTable trial_huf = prev_huf;
                emit_literals(content, literals.data() + rg.lit_a,
                              static_cast<int>(rg.lit_b - rg.lit_a), &trial_huf);
                RepeatOffsets trial_rep = emit_rep;
                FseCTable trial_ll = prev_ll;
                FseCTable trial_of = prev_of;
                FseCTable trial_ml = prev_ml;
                emit_sequences(content, sub_seqs, trial_rep, &trial_ll,
                               &trial_of, &trial_ml, level);
                if (content.size() < sub_raw_len) {
                    emit_block_header(sub_last, 2u,
                                      static_cast<std::uint32_t>(content.size()));
                    output.insert(output.end(), content.begin(), content.end());
                    emit_rep = trial_rep;
                    prev_huf = std::move(trial_huf);
                    prev_ll = std::move(trial_ll);
                    prev_of = std::move(trial_of);
                    prev_ml = std::move(trial_ml);
                    emitted = true;
                }
            }
            if (!emitted) {
                if (is_uniform_bytes(sub_raw, sub_raw_len)) {
                    emit_block_header(sub_last, 1u,
                                      static_cast<std::uint32_t>(sub_raw_len));
                    output.push_back(static_cast<std::byte>(sub_raw[0]));
                } else {
                    emit_block_header(sub_last, 0u,
                                      static_cast<std::uint32_t>(sub_raw_len));
                    const auto* bp = reinterpret_cast<const std::byte*>(sub_raw);
                    output.insert(output.end(), bp, bp + sub_raw_len);
                }
            }
        }

        off = bend;
    }

    if (output.size() >= size) return {};
    return output;
}

}  // namespace fzip::zstd
