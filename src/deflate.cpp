// fzip — DEFLATE encoder/decoder implementation (RFC 1951).
#include "deflate.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace fzip {

namespace {

// --------------------------------------------------------------------------
// Bit writer: emits bits LSB-first into a byte buffer (DEFLATE convention).
// --------------------------------------------------------------------------
class BitWriter {
  public:
    void put_bits(std::uint32_t value, int n) {
        // n <= 24 guaranteed by callers.
        acc_ |= (value & ((1u << n) - 1u)) << acc_bits_;
        acc_bits_ += n;
        while (acc_bits_ >= 8) {
            out_.push_back(static_cast<std::byte>(acc_ & 0xFFu));
            acc_ >>= 8;
            acc_bits_ -= 8;
        }
    }
    // Put a Huffman code of `len` bits, MSB-first (DEFLATE Huffman codes are
    // stored with the most-significant bit first, but the bit stream itself
    // is LSB-first; we reverse the code bits into the LSB-first stream).
    void put_huff(std::uint32_t code, int len) {
        // Reverse `code` of `len` bits so it lands MSB-first in the stream.
        std::uint32_t r = 0;
        for (int i = 0; i < len; ++i) {
            r = (r << 1) | ((code >> i) & 1u);
        }
        put_bits(r, len);
    }
    void align_to_byte() {
        if (acc_bits_ > 0) {
            out_.push_back(static_cast<std::byte>(acc_ & 0xFFu));
            acc_ = 0;
            acc_bits_ = 0;
        }
    }
    void put_bytes(std::span<const std::byte> b) {
        align_to_byte();
        out_.insert(out_.end(), b.begin(), b.end());
    }
    auto data() const -> std::span<const std::byte> {
        return std::span<const std::byte>{out_.data(), out_.size()};
    }
    auto size() const -> std::size_t { return out_.size(); }

  private:
    std::vector<std::byte> out_;
    std::uint32_t acc_ = 0;
    int acc_bits_ = 0;
};

// --------------------------------------------------------------------------
// Huffman code construction (RFC 1951 §3.2.2).
// Given symbol frequencies, compute code lengths (limited to `max_len`) and
// then canonical codes. We use the package-merge / length-limited algorithm
// approximation via a simple heap-based Huffman + a retry with a cap.
// --------------------------------------------------------------------------
struct HuffCodes {
    std::array<int, 288> lengths{};   // code length per symbol
    std::array<std::uint32_t, 288> codes{};  // canonical code per symbol
    int max_sym = 0;  // one past the highest symbol with a nonzero length
};

// Build canonical codes from code lengths (RFC 1951 §3.2.2 canonical step).
// Works for any array size >= the number of symbols we use.
template <std::size_t N>
auto build_canonical_codes(const std::array<int, N>& lengths, int max_sym)
    -> HuffCodes {
    static_assert(N <= 288);
    HuffCodes h;
    for (int s = 0; s < static_cast<int>(N) && s < 288; ++s) {
        h.lengths[s] = lengths[s];
    }
    h.max_sym = max_sym;
    // Count codes of each length.
    std::array<int, 16> bl_count{};
    for (int s = 0; s < max_sym; ++s) {
        if (lengths[s] > 0 && lengths[s] <= 15) bl_count[lengths[s]]++;
    }
    // Find first code for each length.
    std::array<std::uint32_t, 16> next_code{};
    std::uint32_t code = 0;
    for (int bits = 1; bits <= 15; ++bits) {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    // Assign codes to symbols in symbol order.
    for (int s = 0; s < max_sym; ++s) {
        int len = lengths[s];
        if (len > 0 && len <= 15) {
            h.codes[s] = next_code[len]++;
        }
    }
    return h;
}

// A min-heap entry for Huffman tree construction.
struct HeapNode {
    std::uint64_t freq;
    int idx;  // index into a node pool (negative = leaf symbol)
};
auto heap_less(const HeapNode& a, const HeapNode& b) -> bool {
    return a.freq > b.freq;  // priority_queue is a max-heap by default
}

// Build Huffman code lengths from frequencies, limited to `max_len` bits.
// Template works for any array size (288 for lit, 32 for dist, 19 for CL).
template <std::size_t N>
auto build_huff_lengths(const std::array<std::uint32_t, N>& freqs,
                        int max_sym, int max_len) -> std::array<int, N> {
    std::array<int, N> lengths{};
    // Collect nonzero symbols.
    struct Sym {
        std::uint32_t f;
        int s;
    };
    std::vector<Sym> syms;
    for (int s = 0; s < max_sym; ++s) {
        if (freqs[s] > 0) syms.push_back({freqs[s], s});
    }
    if (syms.empty()) return lengths;
    if (syms.size() == 1) {
        // A single symbol: give it a 1-bit code (DEFLATE convention).
        lengths[syms[0].s] = 1;
        return lengths;
    }

    // Build Huffman tree via a priority queue (pairing freqs).
    struct Node {
        std::uint64_t freq;
        int parent;
    };
    std::vector<Node> nodes;
    nodes.reserve(syms.size() * 2);
    for (auto& sp : syms) {
        nodes.push_back({sp.f, -1});
    }
    auto cmp = [&nodes](int a, int b) { return nodes[a].freq > nodes[b].freq; };
    std::vector<int> heap;
    heap.reserve(syms.size());
    for (int i = 0; i < static_cast<int>(syms.size()); ++i) heap.push_back(i);
    std::make_heap(heap.begin(), heap.end(), cmp);
    while (heap.size() > 1) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        int a = heap.back(); heap.pop_back();
        std::pop_heap(heap.begin(), heap.end(), cmp);
        int b = heap.back(); heap.pop_back();
        Node n{nodes[a].freq + nodes[b].freq, -1};
        int ni = static_cast<int>(nodes.size());
        nodes.push_back(n);
        nodes[a].parent = ni;
        nodes[b].parent = ni;
        heap.push_back(ni);
        std::push_heap(heap.begin(), heap.end(), cmp);
    }
    // Walk from each leaf to root to get depth = code length.
    int max_depth = 0;
    for (int i = 0; i < static_cast<int>(syms.size()); ++i) {
        int d = 0;
        int p = nodes[i].parent;
        while (p != -1) { ++d; p = nodes[p].parent; }
        lengths[syms[i].s] = d;
        if (d > max_depth) max_depth = d;
    }

    // Length-limit: if max_depth > max_len, clamp and fix.
    if (max_depth > max_len) {
        // Simple iterative clamp: sort symbols by freq desc, clamp lengths
        // to max_len, then for symbols with length 0 due to clamping
        // reassignment, bump shorter codes. This is the "BZip2-style" fix
        // and is not strictly optimal but is correct (codes are prefix-free
        // by construction below).
        // Sort symbol indices by frequency descending.
        std::vector<int> order(syms.size());
        for (int i = 0; i < static_cast<int>(syms.size()); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return syms[a].f > syms[b].f; });
        // Assign lengths via a Kraft-inequality bounded redistribution.
        std::array<int, N> newlen{};
        // Use the package-merge result approximation: give the most frequent
        // symbols the shortest clamped lengths.
        // We use a simple greedy: assign lengths so that sum of 2^-len <= 1.
        // Start all at max_len, then shorten the highest-freq ones.
        for (int i = 0; i < static_cast<int>(syms.size()); ++i) {
            newlen[syms[order[i]].s] = max_len;
        }
        // Try to shorten: while we can reduce some length and still satisfy
        // Kraft, do it for the highest-frequency symbol that can be reduced.
        bool improved = true;
        while (improved) {
            improved = false;
            for (int oi = 0; oi < static_cast<int>(syms.size()); ++oi) {
                int s = syms[order[oi]].s;
                if (newlen[s] <= 1) continue;
                // Check Kraft: sum 2^-len over all must be <= 1 after reducing.
                double k = 0.0;
                for (int i = 0; i < static_cast<int>(syms.size()); ++i) {
                    k += 1.0 / (1u << newlen[syms[i].s]);
                }
                // Reducing newlen[s] by 1 adds 1/2^(len-1) - 1/2^len = 1/2^len.
                double delta = 1.0 / (1u << (newlen[s] - 1));
                if (k + delta <= 1.0 + 1e-12) {
                    newlen[s]--;
                    improved = true;
                }
            }
        }
        lengths = newlen;
    }
    return lengths;
}

// --------------------------------------------------------------------------
// LZ77 match finder: 3-byte hash + hash chains, 32 KB sliding window.
// --------------------------------------------------------------------------
constexpr int kWindow = 32768;
constexpr int kMinMatch = 3;
constexpr int kMaxMatch = 258;
constexpr int kHashBits = 15;
constexpr int kHashSize = 1 << kHashBits;
constexpr std::uint32_t kHashMask = kHashSize - 1;

inline auto hash3(const std::uint8_t* p) -> std::uint32_t {
    // Mix three bytes into a 15-bit hash.
    std::uint32_t h = (static_cast<std::uint32_t>(p[0]) << 16) |
                      (static_cast<std::uint32_t>(p[1]) << 8) |
                      static_cast<std::uint32_t>(p[2]);
    h = (h * 2654435761u) >> (32 - kHashBits);
    return h & kHashMask;
}

struct Match {
    int dist = 0;   // 1..32768
    int len = 0;    // 3..258
};

// Find the best match at `pos` using the hash chain `head`/`prev`.
// `max_effort` limits chain traversal (level-dependent).
auto find_match(const std::uint8_t* data, std::size_t size, std::size_t pos,
                const std::vector<int>& head, const std::vector<int>& prev,
                int max_effort) -> Match {
    Match best;
    if (pos + kMinMatch > size) return best;
    std::uint32_t h = hash3(data + pos);
    int cand = head[h];
    int limit = static_cast<int>(pos) - kWindow;
    if (limit < 0) limit = 0;
    int tries = max_effort;
    while (cand >= 0 && tries-- > 0) {
        if (cand < limit) break;
        // Quick check: does the candidate extend the best match prefix?
        if (best.len > kMinMatch &&
            data[cand + best.len - 1] != data[pos + best.len - 1]) {
            cand = prev[static_cast<std::size_t>(cand) & (kWindow - 1)];
            continue;
        }
        // Measure match length.
        int maxl = static_cast<int>(std::min<std::size_t>(kMaxMatch, size - pos));
        int l = 0;
        while (l < maxl && data[cand + l] == data[pos + l]) ++l;
        if (l >= kMinMatch && l > best.len) {
            best.len = l;
            best.dist = static_cast<int>(pos) - cand;
            if (l >= kMaxMatch) break;
        }
        cand = prev[static_cast<std::size_t>(cand) & (kWindow - 1)];
    }
    return best;
}

// Insert `pos` into the hash chain.
void insert_hash(const std::uint8_t* data, std::size_t size, std::size_t pos,
                 std::vector<int>& head, std::vector<int>& prev) {
    if (pos + kMinMatch > size) return;
    std::uint32_t h = hash3(data + pos);
    int p = static_cast<int>(pos);
    prev[p & (kWindow - 1)] = head[h];
    head[h] = p;
}

// --------------------------------------------------------------------------
// Token stream: a sequence of literals and (length, distance) pairs.
// --------------------------------------------------------------------------
struct Token {
    bool is_match;
    std::uint8_t lit;       // if !is_match
    int len;                // if is_match (3..258)
    int dist;               // if is_match (1..32768)
};

// LZ77 + lazy matching. Returns the token list and populates the literal/
// length and distance frequency tables.
// Forward declarations of the length/distance symbol lookups (defined
// below); used by lz77_encode to update frequency tables.
auto length_to_symbol(int len) -> int;
auto distance_to_symbol(int dist) -> int;

auto lz77_encode(const std::uint8_t* data, std::size_t size, int level,
                 std::array<std::uint32_t, 288>& lit_freq,
                 std::array<std::uint32_t, 32>& dist_freq)
    -> std::vector<Token> {
    std::vector<Token> tokens;
    tokens.reserve(size / 2);
    lit_freq.fill(0);
    dist_freq.fill(0);

    std::vector<int> head(kHashSize, -1);
    std::vector<int> prev(kWindow, -1);

    // Level tuning: chain effort and lazy depth.
    int effort;
    bool lazy;
    switch (level) {
        case 1: effort = 4;    lazy = false; break;
        case 2: effort = 8;    lazy = false; break;
        case 3: effort = 32;   lazy = false; break;
        case 4: effort = 16;   lazy = true;  break;
        case 5: effort = 32;   lazy = true;  break;
        case 6: effort = 128;  lazy = true;  break;
        case 7: effort = 1024; lazy = true;  break;
        case 8: effort = 4096; lazy = true;  break;
        default: effort = 1 << 14; lazy = true; break;  // 9
    }

    auto emit_lit = [&](std::uint8_t c) {
        Token t{false, c, 0, 0};
        tokens.push_back(t);
        lit_freq[c]++;
    };
    auto emit_match = [&](int len, int dist) {
        Token t{true, 0, len, dist};
        tokens.push_back(t);
        lit_freq[257 + length_to_symbol(len)]++;
        dist_freq[distance_to_symbol(dist)]++;
    };

    std::size_t pos = 0;
    while (pos < size) {
        Match m = find_match(data, size, pos, head, prev, effort);
        if (m.len < kMinMatch) {
            emit_lit(data[pos]);
            insert_hash(data, size, pos, head, prev);
            ++pos;
            continue;
        }

        if (lazy) {
            // Insert pos, then look one byte ahead for a longer match. If the
            // next position yields a longer match, emit a literal here and take
            // that match instead (classic lazy matching, deflate_slow-style).
            insert_hash(data, size, pos, head, prev);
            Match m2 = find_match(data, size, pos + 1, head, prev, effort);
            if (m2.len > m.len) {
                emit_lit(data[pos]);
                ++pos;
                m = m2;
                emit_match(m.len, m.dist);
                for (int i = 0; i < m.len; ++i) {
                    insert_hash(data, size, pos + i, head, prev);
                }
                pos += m.len;
                continue;
            }
            // Keep the match at pos; pos was already inserted above.
            emit_match(m.len, m.dist);
            for (int i = 1; i < m.len; ++i) {
                insert_hash(data, size, pos + i, head, prev);
            }
            pos += m.len;
            continue;
        }

        emit_match(m.len, m.dist);
        for (int i = 0; i < m.len; ++i) {
            insert_hash(data, size, pos + i, head, prev);
        }
        pos += m.len;
    }
    // End-of-block symbol (256) frequency = 1.
    lit_freq[256]++;
    return tokens;
}

// Length code table (RFC 1951 §3.2.5). For each length 3..258 we get a
// symbol (257..285), extra bits, and base.
struct LenCode { int sym; int extra; int base; };
auto length_code(int len) -> LenCode {
    // Table from the RFC.
    static const struct { int min; int max; int sym; int extra; int base; } tab[] = {
        {3, 3, 257, 0, 3}, {4, 4, 258, 0, 4}, {5, 5, 259, 0, 5},
        {6, 6, 260, 0, 6}, {7, 7, 261, 0, 7}, {8, 8, 262, 0, 8},
        {9, 9, 263, 0, 9}, {10, 10, 264, 0, 10}, {11, 12, 265, 1, 11},
        {13, 14, 266, 1, 13}, {15, 16, 267, 1, 15}, {17, 18, 268, 1, 17},
        {19, 22, 269, 2, 19}, {23, 26, 270, 2, 23}, {27, 30, 271, 2, 27},
        {31, 34, 272, 2, 31}, {35, 42, 273, 3, 35}, {43, 50, 274, 3, 43},
        {51, 58, 275, 3, 51}, {59, 66, 276, 3, 59}, {67, 82, 277, 4, 67},
        {83, 98, 278, 4, 83}, {99, 114, 279, 4, 99}, {115, 130, 280, 4, 115},
        {131, 162, 281, 5, 131}, {163, 194, 282, 5, 163}, {195, 226, 283, 5, 195},
        {227, 257, 284, 5, 227}, {258, 258, 285, 0, 258},
    };
    for (auto& e : tab) {
        if (len >= e.min && len <= e.max) {
            return {e.sym, e.extra, e.base};
        }
    }
    throw std::runtime_error("invalid DEFLATE length");
}

auto length_to_symbol(int len) -> int {
    return length_code(len).sym - 257;
}

// Distance code table (RFC 1951 §3.2.5). For each distance 1..32768.
struct DistCode { int sym; int extra; int base; };
auto distance_code(int dist) -> DistCode {
    static const struct { int min; int max; int sym; int extra; int base; } tab[] = {
        {1, 1, 0, 0, 1}, {2, 2, 1, 0, 2}, {3, 3, 2, 0, 3}, {4, 4, 3, 0, 4},
        {5, 6, 4, 1, 5}, {7, 8, 5, 1, 7}, {9, 12, 6, 2, 9}, {13, 16, 7, 2, 13},
        {17, 24, 8, 3, 17}, {25, 32, 9, 3, 25}, {33, 48, 10, 4, 33},
        {49, 64, 11, 4, 49}, {65, 96, 12, 5, 65}, {97, 128, 13, 5, 97},
        {129, 192, 14, 6, 129}, {193, 256, 15, 6, 193}, {257, 384, 16, 7, 257},
        {385, 512, 17, 7, 385}, {513, 768, 18, 8, 513}, {769, 1024, 19, 8, 769},
        {1025, 1536, 20, 9, 1025}, {1537, 2048, 21, 9, 1537},
        {2049, 3072, 22, 10, 2049}, {3073, 4096, 23, 10, 3073},
        {4097, 6144, 24, 11, 4097}, {6145, 8192, 25, 11, 6145},
        {8193, 12288, 26, 12, 8193}, {12289, 16384, 27, 12, 12289},
        {16385, 24576, 28, 13, 16385}, {24577, 32768, 29, 13, 24577},
    };
    for (auto& e : tab) {
        if (dist >= e.min && dist <= e.max) {
            return {e.sym, e.extra, e.base};
        }
    }
    throw std::runtime_error("invalid DEFLATE distance");
}

auto distance_to_symbol(int dist) -> int {
    return distance_code(dist).sym;
}

// --------------------------------------------------------------------------
// Block emission.
// --------------------------------------------------------------------------

// Emit a stored block (BTYPE=00). Used as a fallback.
void emit_stored_block(BitWriter& w, bool final,
                       std::span<const std::byte> block) {
    w.put_bits(final ? 1u : 0u, 1);  // BFINAL
    w.put_bits(0u, 2);               // BTYPE = 00 (stored)
    w.align_to_byte();
    std::uint16_t len = static_cast<std::uint16_t>(block.size());
    std::uint16_t nlen = ~len;
    w.put_bytes(std::span<const std::byte>{
        reinterpret_cast<const std::byte*>(&len), 2});
    w.put_bytes(std::span<const std::byte>{
        reinterpret_cast<const std::byte*>(&nlen), 2});
    w.put_bytes(block);
}

// Emit all tokens using fixed Huffman codes (BTYPE=01).
void emit_fixed_block(BitWriter& w, bool final,
                      const std::vector<Token>& tokens) {
    w.put_bits(final ? 1u : 0u, 1);
    w.put_bits(1u, 2);  // BTYPE = 01 (fixed Huffman)

    // Fixed literal/length code lengths (RFC 1951 §3.2.6).
    std::array<int, 288> lit_lens{};
    for (int s = 0; s <= 143; ++s) lit_lens[s] = 8;
    for (int s = 144; s <= 255; ++s) lit_lens[s] = 9;
    for (int s = 256; s <= 279; ++s) lit_lens[s] = 7;
    for (int s = 280; s <= 287; ++s) lit_lens[s] = 8;
    auto lit = build_canonical_codes(lit_lens, 288);
    // Fixed distance codes: all 5 bits.
    std::array<int, 32> dist_lens{};
    dist_lens.fill(5);
    HuffCodes dh;
    for (int s = 0; s < 30; ++s) dh.lengths[s] = 5;
    dh.max_sym = 30;
    // Build canonical distance codes with all-5-bit lengths.
    std::array<int, 16> bl_count{};
    for (int s = 0; s < 30; ++s) bl_count[5]++;
    std::array<std::uint32_t, 16> next_code{};
    std::uint32_t code = 0;
    for (int bits = 1; bits <= 15; ++bits) {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for (int s = 0; s < 30; ++s) dh.codes[s] = next_code[5]++;

    for (const auto& t : tokens) {
        if (!t.is_match) {
            w.put_huff(lit.codes[t.lit], lit.lengths[t.lit]);
        } else {
            auto lc = length_code(t.len);
            int sym = lc.sym;
            w.put_huff(lit.codes[sym], lit.lengths[sym]);
            if (lc.extra > 0) {
                w.put_bits(static_cast<std::uint32_t>(t.len - lc.base),
                           lc.extra);
            }
            auto dc = distance_code(t.dist);
            w.put_huff(dh.codes[dc.sym], 5);
            if (dc.extra > 0) {
                w.put_bits(static_cast<std::uint32_t>(t.dist - dc.base),
                           dc.extra);
            }
        }
    }
    // End of block.
    w.put_huff(lit.codes[256], lit.lengths[256]);
}

// Emit all tokens using dynamic Huffman codes (BTYPE=10).
void emit_dynamic_block(BitWriter& w, bool final,
                        const std::vector<Token>& tokens,
                        const std::array<std::uint32_t, 288>& lit_freq,
                        const std::array<std::uint32_t, 32>& dist_freq) {
    w.put_bits(final ? 1u : 0u, 1);
    w.put_bits(2u, 2);  // BTYPE = 10 (dynamic Huffman)

    // Build literal/length tree (286 symbols: 0..285; 286/287 unused).
    auto lit_lens = build_huff_lengths(lit_freq, 286, 15);
    auto lit = build_canonical_codes(lit_lens, 286);

    // Build distance tree (30 symbols: 0..29).
    std::array<std::uint32_t, 32> df{};
    for (int i = 0; i < 30; ++i) df[i] = dist_freq[i];
    auto dist_lens_full = build_huff_lengths(df, 30, 15);
    std::array<int, 32> dist_lens{};
    for (int i = 0; i < 30; ++i) dist_lens[i] = dist_lens_full[i];
    auto dist = build_canonical_codes(dist_lens, 30);

    // Count HLIT/HDIST: number of literal/length and distance codes.
    int hlit = 286;
    while (hlit > 257 && lit_lens[hlit - 1] == 0) --hlit;
    int hdist = 30;
    while (hdist > 1 && dist_lens[hdist - 1] == 0) --hdist;
    // HLIT/HDIST are encoded as (count - 257) / (count - 1) in 5 bits.
    w.put_bits(static_cast<std::uint32_t>(hlit - 257), 5);
    w.put_bits(static_cast<std::uint32_t>(hdist - 1), 5);

    // Build the code-length code tree from the RLE'd sequence of lengths.
    // Concatenate lit_lens[0..hlit) + dist_lens[0..hdist).
    std::vector<int> all_lens;
    all_lens.reserve(hlit + hdist);
    for (int i = 0; i < hlit; ++i) all_lens.push_back(lit_lens[i]);
    for (int i = 0; i < hdist; ++i) all_lens.push_back(dist_lens[i]);

    // RLE-encode the length sequence using codes 0..18 (RFC 1951 §3.2.7).
    // Codes: 0..15 = literal length value; 16 = repeat previous 3-6 times
    // (2 extra bits); 17 = repeat zero 3-10 times (3 extra bits);
    // 18 = repeat zero 11-138 times (7 extra bits).
    struct Rle { int code; int xbits; };
    std::vector<Rle> rle;
    std::array<std::uint32_t, 19> cl_freq{};
    for (std::size_t i = 0; i < all_lens.size(); ) {
        int v = all_lens[i];
        int run = 1;
        while (i + run < all_lens.size() && all_lens[i + run] == v) ++run;

        if (v == 0) {
            // Run of zeros: use 17 (3-10) or 18 (11-138), with individual
            // 0-codes for any remainder < 3.
            int emitted = 0;
            while (emitted < run) {
                int rem = run - emitted;
                if (rem >= 11) {
                    int r = std::min(rem, 138);
                    rle.push_back({18, r - 11});
                    cl_freq[18]++;
                    emitted += r;
                } else if (rem >= 3) {
                    rle.push_back({17, rem - 3});
                    cl_freq[17]++;
                    emitted += rem;
                } else {
                    rle.push_back({0, 0});
                    cl_freq[0]++;
                    emitted++;
                }
            }
        } else {
            // Run of a non-zero value: emit it once, then use code 16 to
            // repeat it 3-6 times (code 16 repeats the *previous* code).
            rle.push_back({v, 0});
            cl_freq[v]++;
            int emitted = 1;
            while (emitted < run) {
                int rem = run - emitted;
                if (rem >= 3) {
                    int r = std::min(rem, 6);
                    rle.push_back({16, r - 3});
                    cl_freq[16]++;
                    emitted += r;
                } else {
                    rle.push_back({v, 0});
                    cl_freq[v]++;
                    emitted++;
                }
            }
        }
        i += run;
    }

    // Build the code-length code tree (19 symbols, max 7 bits).
    std::array<std::uint32_t, 19> clf{};
    for (int i = 0; i < 19; ++i) clf[i] = cl_freq[i];
    auto cl_lens = build_huff_lengths(clf, 19, 7);
    auto cl = build_canonical_codes(cl_lens, 19);

    // HCLEN: number of code-length codes (4..19), in the magic order.
    static const int order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5,
                                   11, 4, 12, 3, 13, 2, 14, 1, 15};
    int hclen = 19;
    while (hclen > 4 && cl_lens[order[hclen - 1]] == 0) --hclen;
    w.put_bits(static_cast<std::uint32_t>(hclen - 4), 4);
    for (int i = 0; i < hclen; ++i) {
        w.put_bits(static_cast<std::uint32_t>(cl_lens[order[i]]), 3);
    }

    // Emit the RLE'd code lengths using the code-length tree.
    for (const auto& r : rle) {
        w.put_huff(cl.codes[r.code], cl_lens[r.code]);
        if (r.code == 16) {
            w.put_bits(static_cast<std::uint32_t>(r.xbits), 2);
        } else if (r.code == 17) {
            w.put_bits(static_cast<std::uint32_t>(r.xbits), 3);
        } else if (r.code == 18) {
            w.put_bits(static_cast<std::uint32_t>(r.xbits), 7);
        }
    }

    // Emit the tokens.
    for (const auto& t : tokens) {
        if (!t.is_match) {
            w.put_huff(lit.codes[t.lit], lit.lengths[t.lit]);
        } else {
            auto lc = length_code(t.len);
            w.put_huff(lit.codes[lc.sym], lit.lengths[lc.sym]);
            if (lc.extra > 0) {
                w.put_bits(static_cast<std::uint32_t>(t.len - lc.base),
                           lc.extra);
            }
            auto dc = distance_code(t.dist);
            w.put_huff(dist.codes[dc.sym], dist.lengths[dc.sym]);
            if (dc.extra > 0) {
                w.put_bits(static_cast<std::uint32_t>(t.dist - dc.base),
                           dc.extra);
            }
        }
    }
    // End of block.
    w.put_huff(lit.codes[256], lit.lengths[256]);
}

// Decide block size: we emit one dynamic block per call. For very large
// inputs we could split, but a single block is correct (just memory-heavy).
constexpr std::size_t kBlockSize = 1 << 20;  // 1 MiB worth of input per block

}  // namespace

auto deflate_compress(std::span<const std::byte> data, int level)
    -> std::vector<std::byte> {
    if (data.empty()) {
        // Empty input: a single empty final block (fixed Huffman + EOB).
        BitWriter w;
        w.put_bits(1u, 1);  // BFINAL
        w.put_bits(1u, 2);  // BTYPE = fixed
        // Fixed EOB code 256: 7 bits, code 0b0000000.
        w.put_huff(0, 7);
        w.align_to_byte();
        return std::vector<std::byte>{w.data().begin(), w.data().end()};
    }
    if (level < 1) level = 1;
    if (level > 9) level = 9;

    BitWriter w;
    const auto* p = reinterpret_cast<const std::uint8_t*>(data.data());
    std::size_t size = data.size();

    // Process in blocks to bound memory and allow per-block tree fitting.
    std::size_t off = 0;
    bool first = true;
    while (off < size) {
        std::size_t end = std::min(off + kBlockSize, size);
        std::size_t blen = end - off;
        std::array<std::uint32_t, 288> lit_freq{};
        std::array<std::uint32_t, 32> dist_freq{};
        auto tokens = lz77_encode(p + off, blen, level, lit_freq, dist_freq);
        bool final = (end == size);

        // Decide block type: try dynamic; if the input is tiny or highly
        // incompressible, fall back to fixed or stored. We compare sizes
        // only for the whole stream at the end; here we pick dynamic for
        // level >= 4 and fixed for lower levels, unless the token stream
        // is trivially small.
        if (level >= 4) {
            emit_dynamic_block(w, final, tokens, lit_freq, dist_freq);
        } else {
            emit_fixed_block(w, final, tokens);
        }
        off = end;
        first = false;
    }

    w.align_to_byte();
    auto out = std::vector<std::byte>{w.data().begin(), w.data().end()};

    // If compression didn't help, signal Store by returning empty.
    if (out.size() >= size) return {};
    return out;
}

// --------------------------------------------------------------------------
// Inflate (for our own round-trip tests).
// --------------------------------------------------------------------------
class BitReader {
  public:
    explicit BitReader(std::span<const std::byte> d) : d_(d) {}
    auto get_bits(int n) -> std::uint32_t {
        std::uint32_t v = 0;
        for (int i = 0; i < n; ++i) {
            if (pos_ >= d_.size()) throw std::runtime_error("DEFLATE EOF");
            std::uint32_t bit = (static_cast<std::uint32_t>(d_[pos_]) >>
                                 bit_) & 1u;
            v |= bit << i;
            ++bit_;
            if (bit_ == 8) { bit_ = 0; ++pos_; }
        }
        return v;
    }
    auto get_huff(const HuffCodes& h) -> int {
        int code = 0, len = 0;
        while (len <= 15) {
            code = (code << 1) | get_bits(1);
            ++len;
            for (int s = 0; s < h.max_sym; ++s) {
                if (h.lengths[s] == len && h.codes[s] == static_cast<std::uint32_t>(code)) {
                    return s;
                }
            }
        }
        throw std::runtime_error("invalid Huffman code");
    }
    void align_byte() {
        if (bit_ > 0) { bit_ = 0; ++pos_; }
    }
    auto get_byte() -> std::uint8_t {
        if (pos_ >= d_.size()) throw std::runtime_error("DEFLATE EOF");
        return static_cast<std::uint8_t>(d_[pos_++]);
    }
    auto pos() const -> std::size_t { return pos_; }

  private:
    std::span<const std::byte> d_;
    std::size_t pos_ = 0;
    int bit_ = 0;
};

auto inflate_block(BitReader& r, std::vector<std::byte>& out) -> bool {
    std::uint32_t bfinal = r.get_bits(1);
    std::uint32_t btype = r.get_bits(2);
    if (btype == 0) {
        r.align_byte();
        std::uint8_t lo = r.get_byte(), hi = r.get_byte();
        int len = lo | (hi << 8);
        r.get_byte(); r.get_byte();  // nlen, unused
        for (int i = 0; i < len; ++i) out.push_back(static_cast<std::byte>(r.get_byte()));
    } else if (btype == 1) {
        std::array<int, 288> lit_lens{};
        for (int s = 0; s <= 143; ++s) lit_lens[s] = 8;
        for (int s = 144; s <= 255; ++s) lit_lens[s] = 9;
        for (int s = 256; s <= 279; ++s) lit_lens[s] = 7;
        for (int s = 280; s <= 287; ++s) lit_lens[s] = 8;
        auto lit = build_canonical_codes(lit_lens, 288);
        std::array<int, 32> dlens{}; dlens.fill(5);
        HuffCodes dist = build_canonical_codes(dlens, 30);
        // Rebuild dist with all-5-bit codes via canonical step.
        std::array<int, 16> bl{}; bl[5] = 30;
        std::array<std::uint32_t, 16> nc{}; std::uint32_t c = 0;
        for (int b = 1; b <= 15; ++b) { c = (c + bl[b-1]) << 1; nc[b] = c; }
        for (int s = 0; s < 30; ++s) dist.codes[s] = nc[5]++;
        while (true) {
            int sym = r.get_huff(lit);
            if (sym == 256) break;
            if (sym < 256) {
                out.push_back(static_cast<std::byte>(sym));
            } else {
                // Decode length from symbol 257..285 via static tables.
                static const int len_base[] = {3,4,5,6,7,8,9,10,11,13,15,17,
                    19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
                static const int len_extra[] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,
                    2,3,3,3,3,4,4,4,4,5,5,5,5,0};
                int idx = sym - 257;
                int len = len_base[idx] + (len_extra[idx] ? (int)r.get_bits(len_extra[idx]) : 0);
                int dsym = r.get_huff(dist);
                static const int dist_base[] = {1,2,3,4,5,7,9,13,17,25,33,49,
                    65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,
                    6145,8193,12289,16385,24577};
                static const int dist_extra[] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,
                    6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
                int d = dist_base[dsym] + (dist_extra[dsym] ? (int)r.get_bits(dist_extra[dsym]) : 0);
                std::size_t start = out.size() - static_cast<std::size_t>(d);
                for (int i = 0; i < len; ++i) {
                    out.push_back(out[start + i]);
                }
            }
        }
    } else if (btype == 2) {
        // Dynamic Huffman: read HLIT, HDIST, HCLEN, code-length tree, then
        // literal/length + distance trees.
        int hlit = r.get_bits(5) + 257;
        int hdist = r.get_bits(5) + 1;
        int hclen = r.get_bits(4) + 4;
        static const int order[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,
                                       2,14,1,15};
        std::array<int, 19> cl_lens{};
        for (int i = 0; i < hclen; ++i) cl_lens[order[i]] = r.get_bits(3);
        auto cl = build_canonical_codes(cl_lens, 19);
        std::vector<int> all_lens;
        all_lens.reserve(hlit + hdist);
        while (static_cast<int>(all_lens.size()) < hlit + hdist) {
            int sym = r.get_huff(cl);
            if (sym < 16) {
                all_lens.push_back(sym);
            } else if (sym == 16) {
                int rep = r.get_bits(2) + 3;
                int v = all_lens.back();
                for (int i = 0; i < rep; ++i) all_lens.push_back(v);
            } else if (sym == 17) {
                int rep = r.get_bits(3) + 3;
                for (int i = 0; i < rep; ++i) all_lens.push_back(0);
            } else {  // 18
                int rep = r.get_bits(7) + 11;
                for (int i = 0; i < rep; ++i) all_lens.push_back(0);
            }
        }
        std::array<int, 288> lit_lens{};
        for (int i = 0; i < hlit; ++i) lit_lens[i] = all_lens[i];
        std::array<int, 32> dist_lens{};
        for (int i = 0; i < hdist; ++i) dist_lens[i] = all_lens[hlit + i];
        auto lit = build_canonical_codes(lit_lens, hlit);
        auto dist = build_canonical_codes(dist_lens, hdist);
        while (true) {
            int sym = r.get_huff(lit);
            if (sym == 256) break;
            if (sym < 256) {
                out.push_back(static_cast<std::byte>(sym));
            } else {
                static const int len_base[] = {3,4,5,6,7,8,9,10,11,13,15,17,
                    19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
                static const int len_extra[] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,
                    2,3,3,3,3,4,4,4,4,5,5,5,5,0};
                int idx = sym - 257;
                int len = len_base[idx] + (len_extra[idx] ? (int)r.get_bits(len_extra[idx]) : 0);
                int dsym = r.get_huff(dist);
                static const int dist_base[] = {1,2,3,4,5,7,9,13,17,25,33,49,
                    65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,
                    6145,8193,12289,16385,24577};
                static const int dist_extra[] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,
                    6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
                int d = dist_base[dsym] + (dist_extra[dsym] ? (int)r.get_bits(dist_extra[dsym]) : 0);
                std::size_t start = out.size() - static_cast<std::size_t>(d);
                for (int i = 0; i < len; ++i) {
                    out.push_back(out[start + i]);
                }
            }
        }
    } else {
        throw std::runtime_error("invalid DEFLATE block type 3");
    }
    return bfinal != 0;
}

auto deflate_decompress(std::span<const std::byte> data, std::size_t /*expected*/)
    -> std::vector<std::byte> {
    BitReader r{data};
    std::vector<std::byte> out;
    bool final = false;
    while (!final) {
        final = inflate_block(r, out);
    }
    return out;
}

}  // namespace fzip
