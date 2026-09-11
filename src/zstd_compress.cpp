// fzip — zstd compressor implementation (RFC 8878).
//
// Emits spec-compliant compressed blocks: a raw literals section followed by
// a sequences section encoded with the predefined FSE tables (symbol
// compression mode 0). LZ77 uses a greedy hash-chain match finder over a
// 32 KiB window, matching the frame's declared window.
#include "zstd.hpp"
#include "zstd_internal.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>

#include "xxhash.hpp"
#include "zstd_fse.hpp"
#include "zstd_huffman.hpp"
#include "zstd_predefined.hpp"
#include "zstd_sequence.hpp"

namespace fzip::zstd {

namespace {

// --- LZ77 match finder: 3-byte hash + hash chains over a configurable
// window. Chains persist across blocks so matches can reach into earlier
// blocks (bounded by `window`). ---
constexpr int kHashBits = 16;
constexpr int kHashSize = 1 << kHashBits;
constexpr int kMinMatch = 3;
constexpr int kMaxMatch = 131074;  // zstd max match length

// Block content (and decompressed size) is capped by min(window, 128 KiB).
constexpr int kBlockSize = 131072;

struct Match {
    int distance = 0;
    int length = 0;
};

inline auto hash3(const std::uint8_t* p) -> std::uint32_t {
    std::uint32_t h = (static_cast<std::uint32_t>(p[0]) << 16) |
                      (static_cast<std::uint32_t>(p[1]) << 8) |
                      static_cast<std::uint32_t>(p[2]);
    return (h * 2654435761u) >> (32 - kHashBits);
}

struct MatchFinder {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    int window = kBlockSize;  // power of two
    int mask = kBlockSize - 1;
    std::vector<int> head;  // kHashSize
    std::vector<int> prev;  // window

    void reset(const std::uint8_t* d, std::size_t n, int w) {
        data = d;
        size = n;
        window = w;
        mask = w - 1;
        head.assign(kHashSize, -1);
        prev.assign(static_cast<std::size_t>(w), -1);
    }

    void insert(std::size_t pos) {
        if (pos + kMinMatch > size) return;
        std::uint32_t h = hash3(data + pos);
        int p = static_cast<int>(pos);
        prev[static_cast<std::size_t>(p) & static_cast<std::size_t>(mask)] =
            head[h];
        head[h] = p;
    }

    // Find the longest match at `pos` that ends no later than `end`.
    auto find(std::size_t pos, std::size_t end, int effort) const -> Match {
        Match best;
        if (pos + kMinMatch > end) return best;
        const int maxl = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(kMaxMatch), end - pos));
        // Stop walking the chain once the match is "good enough": further
        // search rarely beats it and repetitive data would otherwise make
        // each comparison run to the end of the block for every candidate.
        int nice_len = effort < 64 ? 64 : (effort > 258 ? 258 : effort);
        if (nice_len > maxl) nice_len = maxl;
        std::uint32_t h = hash3(data + pos);
        int cand = head[h];
        int limit = static_cast<int>(pos) - window;
        if (limit < 0) limit = 0;
        int tries = effort;
        while (cand >= 0 && tries-- > 0) {
            if (cand < limit) break;
            if (best.length > kMinMatch &&
                data[cand + best.length - 1] != data[pos + best.length - 1]) {
                cand = prev[static_cast<std::size_t>(cand) &
                            static_cast<std::size_t>(mask)];
                continue;
            }
            int l = 0;
            while (l < maxl && data[cand + l] == data[pos + l]) ++l;
            if (l >= kMinMatch && l > best.length) {
                best.length = l;
                best.distance = static_cast<int>(pos) - cand;
                if (l >= maxl || l >= nice_len) break;
            }
            cand = prev[static_cast<std::size_t>(cand) &
                        static_cast<std::size_t>(mask)];
        }
        return best;
    }
};

// One sequence: `lit_len` literals, then a match of `match_len` at
// `match_dist` bytes back.
struct Seq {
    int lit_len = 0;
    int match_len = 0;
    int match_dist = 0;
};

// LZ77 + lazy matching over `[start, end)`. Positions are absolute within
// `mf.data`; the shared finder may reference earlier blocks.
auto build_sequences(MatchFinder& mf, std::size_t start, std::size_t end,
                     int effort, bool lazy,
                     std::vector<std::uint8_t>& literals) -> std::vector<Seq> {
    std::vector<Seq> seqs;
    literals.clear();

    auto flush_literals = [&](std::size_t from, std::size_t to) {
        for (std::size_t i = from; i < to; ++i) {
            literals.push_back(mf.data[i]);
        }
    };

    std::size_t pos = start;
    std::size_t lit_start = start;
    while (pos < end) {
        Match m = mf.find(pos, end, effort);
        if (m.length < kMinMatch) {
            mf.insert(pos);
            ++pos;
            continue;
        }

        // Lazy matching: insert pos, then look one byte ahead. If the next
        // position yields a longer match, emit a literal here and take it.
        if (lazy && pos + 1 < end) {
            mf.insert(pos);
            Match m2 = mf.find(pos + 1, end, effort);
            if (m2.length > m.length) {
                ++pos;
                m = m2;
                flush_literals(lit_start, pos);
                Seq s;
                s.lit_len = static_cast<int>(pos - lit_start);
                s.match_len = m.length;
                s.match_dist = m.distance;
                seqs.push_back(s);
                for (int i = 0; i < m.length; ++i) {
                    mf.insert(pos + static_cast<std::size_t>(i));
                }
                pos += static_cast<std::size_t>(m.length);
                lit_start = pos;
                continue;
            }
        } else {
            mf.insert(pos);
        }

        // Take the match at pos (pos already inserted).
        flush_literals(lit_start, pos);
        Seq s;
        s.lit_len = static_cast<int>(pos - lit_start);
        s.match_len = m.length;
        s.match_dist = m.distance;
        seqs.push_back(s);
        for (int i = 1; i < m.length; ++i) {
            mf.insert(pos + static_cast<std::size_t>(i));
        }
        pos += static_cast<std::size_t>(m.length);
        lit_start = pos;
    }
    flush_literals(lit_start, end);
    return seqs;
}

struct Code {
    int code = 0;
    int extra_bits = 0;
    int extra_val = 0;
};

// Shannon cost in bits for a symbol with the given frequency.
struct Code;
auto lit_len_code(int len) -> Code;
auto match_len_code(int len) -> Code;
auto offset_code(int dist) -> Code;

inline auto sym_cost(int freq, int total) -> float {
    if (freq <= 0 || total <= 0) return 12.0f;
    return -std::log2(static_cast<float>(freq) / static_cast<float>(total));
}

// Optimal (shortest-path) LZ77 parse over `[start, end)`. A single forward
// pass records the longest match at every position (extending the shared
// finder). Symbol costs are refined over a few DP iterations: a greedy parse
// seeds the statistics, then each pass updates them from the new parse.
auto build_sequences_optimal(MatchFinder& mf, std::size_t start,
                             std::size_t end, int effort, int passes,
                             std::vector<std::uint8_t>& literals)
    -> std::vector<Seq> {
    const int N = static_cast<int>(end - start);
    if (N <= 0) {
        literals.clear();
        return {};
    }

    // Forward pass: longest match at every position.
    std::vector<int> best_len(static_cast<std::size_t>(N), 0);
    std::vector<int> best_dist(static_cast<std::size_t>(N), 0);
    for (int p = 0; p < N; ++p) {
        const std::size_t pos = start + static_cast<std::size_t>(p);
        Match m = mf.find(pos, end, effort);
        best_len[static_cast<std::size_t>(p)] = m.length;
        best_dist[static_cast<std::size_t>(p)] = m.distance;
        mf.insert(pos);
    }

    // Max match length representable by each match-length code.
    int ml_range_max[53];
    for (int c = 0; c < 53; ++c) {
        ml_range_max[c] = matchlen_code_to_base(c) +
                          (1 << matchlen_code_to_extra(c)) - 1;
    }
    const int c_min = match_len_code(kMinMatch).code;

    // Symbol statistics, seeded by a greedy parse over the recorded matches.
    int lit_freq[256] = {0};
    int ll_freq[36] = {0};
    int of_freq[32] = {0};
    int ml_freq[53] = {0};
    int n_lit = 0;
    int n_seq = 0;
    {
        int p = 0;
        int lit_run = 0;
        while (p < N) {
            const int L = best_len[static_cast<std::size_t>(p)];
            if (L >= kMinMatch) {
                ll_freq[lit_len_code(lit_run).code]++;
                of_freq[offset_code(
                            best_dist[static_cast<std::size_t>(p)]).code]++;
                ml_freq[match_len_code(L).code]++;
                ++n_seq;
                p += L;
                lit_run = 0;
            } else {
                lit_freq[mf.data[start + static_cast<std::size_t>(p)]]++;
                ++n_lit;
                ++p;
                ++lit_run;
            }
        }
    }

    std::vector<float> cost(static_cast<std::size_t>(N) + 1, 0.0f);
    std::vector<int> choice_len(static_cast<std::size_t>(N), 1);
    std::vector<int> choice_dist(static_cast<std::size_t>(N), 0);
    std::vector<Seq> seqs;
    literals.clear();

    if (passes < 1) passes = 1;
    for (int pass = 0; pass < passes; ++pass) {
        float lit_cost[256];
        for (int s = 0; s < 256; ++s) {
            float c = sym_cost(lit_freq[s], n_lit);
            if (c > 12.0f) c = 12.0f;
            if (c < 1.0f) c = 1.0f;
            lit_cost[s] = c;
        }
        float mlc_cost[53];
        for (int c = 0; c < 53; ++c) {
            mlc_cost[c] = sym_cost(ml_freq[c], n_seq) +
                          static_cast<float>(matchlen_code_to_extra(c));
        }
        float ofc_cost[32];
        for (int c = 0; c < 32; ++c) {
            ofc_cost[c] = sym_cost(of_freq[c], n_seq) + static_cast<float>(c);
        }

        // Backward DP.
        cost[static_cast<std::size_t>(N)] = 0.0f;
        for (int i = N - 1; i >= 0; --i) {
            float best =
                lit_cost[mf.data[start + static_cast<std::size_t>(i)]] +
                cost[static_cast<std::size_t>(i) + 1];
            int bl = 1;
            int bd = 0;
            const int ml = best_len[static_cast<std::size_t>(i)];
            if (ml >= kMinMatch) {
                const int d = best_dist[static_cast<std::size_t>(i)];
                const float dc = ofc_cost[offset_code(d).code];
                const int c_max = match_len_code(ml).code;
                for (int c = c_min; c <= c_max; ++c) {
                    int L = ml_range_max[c];
                    if (L > ml) L = ml;
                    if (L < kMinMatch) continue;
                    float mc = mlc_cost[c] + dc +
                               cost[static_cast<std::size_t>(i + L)];
                    if (mc < best) {
                        best = mc;
                        bl = L;
                        bd = d;
                    }
                }
            }
            cost[static_cast<std::size_t>(i)] = best;
            choice_len[static_cast<std::size_t>(i)] = bl;
            choice_dist[static_cast<std::size_t>(i)] = bd;
        }

        // Reconstruct the token stream.
        seqs.clear();
        literals.clear();
        int pos = 0;
        int lit_start = 0;
        while (pos < N) {
            const int L = choice_len[static_cast<std::size_t>(pos)];
            if (L <= 1) {
                ++pos;
                continue;
            }
            for (int i = lit_start; i < pos; ++i) {
                literals.push_back(
                    mf.data[start + static_cast<std::size_t>(i)]);
            }
            Seq s;
            s.lit_len = pos - lit_start;
            s.match_len = L;
            s.match_dist = choice_dist[static_cast<std::size_t>(pos)];
            seqs.push_back(s);
            pos += L;
            lit_start = pos;
        }
        for (int i = lit_start; i < N; ++i) {
            literals.push_back(mf.data[start + static_cast<std::size_t>(i)]);
        }

        // Recompute statistics from this parse for the next iteration.
        if (pass + 1 < passes) {
            std::fill_n(lit_freq, 256, 0);
            std::fill_n(ll_freq, 36, 0);
            std::fill_n(of_freq, 32, 0);
            std::fill_n(ml_freq, 53, 0);
            for (std::uint8_t b : literals) lit_freq[b]++;
            n_lit = static_cast<int>(literals.size());
            for (const Seq& s : seqs) {
                ll_freq[lit_len_code(s.lit_len).code]++;
                of_freq[offset_code(s.match_dist).code]++;
                ml_freq[match_len_code(s.match_len).code]++;
            }
            n_seq = static_cast<int>(seqs.size());
        }
    }
    return seqs;
}

// --- Code tables (RFC 8878 §4.2.2) ---

Code lit_len_code(int len) {
    static const int kBase[36] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048,
        4096, 8192, 16384, 32768, 65536};
    static const int kExtra[36] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    for (int c = 35; c >= 0; --c) {
        if (len >= kBase[c]) return {c, kExtra[c], len - kBase[c]};
    }
    return {0, 0, 0};
}

Code match_len_code(int len) {
    static const int kBase[53] = {
        3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027,
        2051, 4099, 8195, 16387, 32771, 65539};
    static const int kExtra[53] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    for (int c = 52; c >= 0; --c) {
        if (len >= kBase[c]) return {c, kExtra[c], len - kBase[c]};
    }
    return {0, 0, 0};
}

// Offset code: Offset_Value = (1 << code) + extra, offset = Offset_Value - 3.
// Codes 0..1 are repeat codes; explicit offsets use code >= 2.
Code offset_code(int dist) {
    int v = dist + 3;
    int code = 0;
    while ((1 << (code + 1)) <= v) ++code;
    if (code < 2) code = 2;  // never use repeat codes
    return {code, code, v - (1 << code)};
}

// Pick the cheapest offset code for `dist` given the current repeat state,
// updating `rep` to exactly the state the decoder will have afterwards.
// Mirrors ZSTD_decodeSequence's repeat-offset logic (RFC 8878 §3.1.1.5).
Code choose_offset(int dist, int lit_len, RepeatOffsets& rep) {
    const bool ll0 = (lit_len == 0);
    if (!ll0) {
        if (dist == rep.offsets[0]) {
            return {0, 0, 0};  // RO1: code 0, state unchanged
        }
        if (dist == rep.offsets[1]) {
            int r = rep.offsets[1];
            rep.offsets[1] = rep.offsets[0];
            rep.offsets[0] = r;
            return {1, 1, 0};  // RO2: code 1, extra bit 0
        }
        if (dist == rep.offsets[2]) {
            int r = rep.offsets[2];
            rep.offsets[2] = rep.offsets[1];
            rep.offsets[1] = rep.offsets[0];
            rep.offsets[0] = r;
            return {1, 1, 1};  // RO3: code 1, extra bit 1
        }
    } else {
        if (dist == rep.offsets[1]) {
            int r = rep.offsets[1];
            rep.offsets[1] = rep.offsets[0];
            rep.offsets[0] = r;
            return {0, 0, 0};  // RO2: code 0 (ll0 shifts)
        }
        if (dist == rep.offsets[2]) {
            int r = rep.offsets[2];
            rep.offsets[2] = rep.offsets[1];
            rep.offsets[1] = rep.offsets[0];
            rep.offsets[0] = r;
            return {1, 1, 0};  // RO3: code 1, extra bit 0
        }
        if (rep.offsets[0] > 1 && dist == rep.offsets[0] - 1) {
            int newv = rep.offsets[0] - 1;
            rep.offsets[2] = rep.offsets[1];
            rep.offsets[1] = rep.offsets[0];
            rep.offsets[0] = newv;
            return {1, 1, 1};  // RO1-1: code 1, extra bit 1
        }
    }
    // Explicit offset.
    Code c = offset_code(dist);
    rep.offsets[2] = rep.offsets[1];
    rep.offsets[1] = rep.offsets[0];
    rep.offsets[0] = dist;
    return c;
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

const FseCTable& ll_ctable() {
    static const FseCTable ct = build_fse_ctable(6, kLLNorm, 35);
    return ct;
}
const FseCTable& of_ctable() {
    static const FseCTable ct = build_fse_ctable(5, kOFNorm, 28);
    return ct;
}
const FseCTable& ml_ctable() {
    static const FseCTable ct = build_fse_ctable(6, kMLNorm, 52);
    return ct;
}

// --- Section emission ---
// Write a little-endian bitfield header for the literals section.
void put_literals_header(std::vector<std::byte>& out, int type, int size_format,
                         int regen, int csize, int k) {
    std::uint64_t v = static_cast<std::uint64_t>(type) |
                      (static_cast<std::uint64_t>(size_format) << 2) |
                      (static_cast<std::uint64_t>(regen) << 4) |
                      (static_cast<std::uint64_t>(csize) << (4 + k));
    const int nbytes = (4 + 2 * k) / 8;
    for (int i = 0; i < nbytes; ++i) {
        out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
    }
}

// Emit the literals section, choosing Huffman (Compressed), RLE, or Raw.
void emit_literals(std::vector<std::byte>& out,
                   const std::vector<std::uint8_t>& lits) {
    const int n = static_cast<int>(lits.size());

    // Try Huffman-coded literals (4-stream).
    if (n >= 64) {
        std::vector<std::byte> body;
        if (huf_compress_literals(lits.data(), n, body)) {
            const int m = std::max(n, static_cast<int>(body.size()));
            int size_format;
            int k;
            if (m <= 1023) {
                size_format = 1; k = 10;
            } else if (m <= 16383) {
                size_format = 2; k = 14;
            } else {
                size_format = 3; k = 18;
            }
            const int hdr_bytes = (4 + 2 * k) / 8;
            if (static_cast<int>(body.size()) + hdr_bytes < n) {
                put_literals_header(out, 2, size_format, n,
                                    static_cast<int>(body.size()), k);
                out.insert(out.end(), body.begin(), body.end());
                return;
            }
        }
    }

    // RLE literals.
    bool all_same = n > 0;
    for (int i = 1; i < n; ++i) {
        if (lits[static_cast<std::size_t>(i)] != lits[0]) {
            all_same = false;
            break;
        }
    }
    if (all_same) {
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

    // Raw literals.
    if (n <= 31) {
        // 1-byte header, Size_Format 0, type 0 (Raw).
        out.push_back(static_cast<std::byte>((n << 3)));
    } else if (n <= 4095) {
        // 2-byte header, Size_Format 1.
        out.push_back(static_cast<std::byte>(((n & 0xF) << 4) | (1 << 2)));
        out.push_back(static_cast<std::byte>((n >> 4) & 0xFF));
    } else {
        // 3-byte header, Size_Format 3.
        out.push_back(static_cast<std::byte>(((n & 0xF) << 4) | (3 << 2)));
        out.push_back(static_cast<std::byte>((n >> 4) & 0xFF));
        out.push_back(static_cast<std::byte>((n >> 12) & 0xFF));
    }
    for (std::uint8_t b : lits) out.push_back(static_cast<std::byte>(b));
}

// A per-stream sequence FSE table and its symbol-compression mode.
struct SeqStream {
    int mode = 0;  // 0 predefined, 1 RLE, 2 FSE
    int rle_sym = 0;
    FseCTable ct;
    std::vector<std::byte> desc;  // NCount table description (mode 2)
};

auto make_seq_stream(const int* freq, int max_code, int max_log, int n,
                     const FseCTable& predef) -> SeqStream {
    SeqStream s;
    int present = 0;
    int only = 0;
    for (int c = 0; c <= max_code; ++c) {
        if (freq[c] > 0) {
            ++present;
            only = c;
        }
    }
    if (present <= 1) {
        s.mode = 1;  // RLE
        s.rle_sym = only;
        return s;
    }
    if (n < 32) {
        s.mode = 0;  // predefined (table description would cost more)
        s.ct = predef;
        return s;
    }
    s.mode = 2;  // inline FSE table
    std::vector<unsigned> counts(static_cast<std::size_t>(max_code) + 1);
    for (int c = 0; c <= max_code; ++c) {
        counts[static_cast<std::size_t>(c)] =
            static_cast<unsigned>(freq[c]);
    }
    std::vector<int> norm;
    fse_normalize(counts.data(), n, max_code, max_log, norm);
    std::vector<std::int16_t> norm16(static_cast<std::size_t>(max_code) + 1);
    for (int c = 0; c <= max_code; ++c) {
        norm16[static_cast<std::size_t>(c)] =
            static_cast<std::int16_t>(norm[static_cast<std::size_t>(c)]);
    }
    s.ct = build_fse_ctable(max_log, norm16.data(), max_code);
    fse_write_ncount(s.desc, norm.data(), max_code, max_log);
    return s;
}

void emit_seq_table(std::vector<std::byte>& out, const SeqStream& s) {
    if (s.mode == 1) {
        out.push_back(static_cast<std::byte>(s.rle_sym));
    } else if (s.mode == 2) {
        out.insert(out.end(), s.desc.begin(), s.desc.end());
    }
}

void emit_sequences(std::vector<std::byte>& out, const std::vector<Seq>& seqs,
                    RepeatOffsets& rep) {
    const int n = static_cast<int>(seqs.size());
    if (n == 0) {
        out.push_back(std::byte{0});
        return;
    }
    // Number_of_Sequences.
    if (n < 128) {
        out.push_back(static_cast<std::byte>(n));
    } else if (n <= 0x7EFF) {
        out.push_back(static_cast<std::byte>(128 + (n >> 8)));
        out.push_back(static_cast<std::byte>(n & 0xFF));
    } else {
        int v = n - 0x7F00;
        out.push_back(static_cast<std::byte>(255));
        out.push_back(static_cast<std::byte>(v & 0xFF));
        out.push_back(static_cast<std::byte>((v >> 8) & 0xFF));
    }

    // Precompute codes and per-block frequencies. Offset codes are chosen in
    // sequence order (tracking repeat offsets) so rep codes 0/1 can be used.
    std::vector<int> llc(static_cast<std::size_t>(n));
    std::vector<int> mlc(static_cast<std::size_t>(n));
    std::vector<Code> ofcodes(static_cast<std::size_t>(n));
    int ll_freq[36] = {0};
    int of_freq[32] = {0};
    int ml_freq[53] = {0};
    RepeatOffsets rep_next = rep;
    for (int i = 0; i < n; ++i) {
        const Seq& s = seqs[static_cast<std::size_t>(i)];
        llc[static_cast<std::size_t>(i)] = lit_len_code(s.lit_len).code;
        ofcodes[static_cast<std::size_t>(i)] =
            choose_offset(s.match_dist, s.lit_len, rep_next);
        mlc[static_cast<std::size_t>(i)] = match_len_code(s.match_len).code;
        ll_freq[llc[static_cast<std::size_t>(i)]]++;
        of_freq[ofcodes[static_cast<std::size_t>(i)].code]++;
        ml_freq[mlc[static_cast<std::size_t>(i)]]++;
    }
    rep = rep_next;
    SeqStream ll = make_seq_stream(ll_freq, 35, 9, n, ll_ctable());
    SeqStream of = make_seq_stream(of_freq, 31, 8, n, of_ctable());
    SeqStream ml = make_seq_stream(ml_freq, 52, 9, n, ml_ctable());

    // Symbol compression modes + table descriptions (LL, OF, ML order).
    out.push_back(static_cast<std::byte>((ll.mode << 6) | (of.mode << 4) |
                                         (ml.mode << 2)));
    emit_seq_table(out, ll);
    emit_seq_table(out, of);
    emit_seq_table(out, ml);

    FseBitWriter w;
    std::uint32_t ll_state = 0, of_state = 0, ml_state = 0;

    // Last sequence: its FSE symbols live in the initial states; only its
    // extra bits are written here.
    {
        const Seq& s = seqs[static_cast<std::size_t>(n - 1)];
        Code lc = lit_len_code(s.lit_len);
        Code mc = match_len_code(s.match_len);
        Code oc = ofcodes[static_cast<std::size_t>(n - 1)];
        if (ll.mode != 1) ll_state = fse_init_cstate2(ll.ct, lc.code);
        if (of.mode != 1) of_state = fse_init_cstate2(of.ct, oc.code);
        if (ml.mode != 1) ml_state = fse_init_cstate2(ml.ct, mc.code);
        if (lc.extra_bits) w.put_bits(static_cast<std::uint32_t>(lc.extra_val), lc.extra_bits);
        if (mc.extra_bits) w.put_bits(static_cast<std::uint32_t>(mc.extra_val), mc.extra_bits);
        if (oc.extra_bits) w.put_bits(static_cast<std::uint32_t>(oc.extra_val), oc.extra_bits);
    }
    for (int i = n - 2; i >= 0; --i) {
        const Seq& s = seqs[static_cast<std::size_t>(i)];
        Code lc = lit_len_code(s.lit_len);
        Code mc = match_len_code(s.match_len);
        Code oc = ofcodes[static_cast<std::size_t>(i)];
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

}  // namespace

auto compress(std::span<const std::byte> data, int level,
              int /*long_distance_log*/) -> std::vector<std::byte> {
    if (data.empty()) return {};
    if (level < 1) level = 1;
    if (level > 22) level = 22;
    // Map zstd level to a match-finder effort.
    int effort = 1 << std::min(level, 10);  // cap chain walks at 1024 steps
    // Lazy matching roughly doubles match-finder work, so only enable it for
    // levels where ratio matters more than throughput.
    bool lazy = (level >= 6);
    // Optimal (shortest-path) parsing for the highest levels.
    bool optimal = (level >= 19);
    int passes = (level >= 22) ? 4 : 3;
    // Levels >= 10 use a window that spans multiple blocks, so matches can
    // reach into earlier blocks (the decoder is sequential and supports it).
    // Faster levels keep independent per-block matching.
    const bool windowed = (level >= 10);
    const int window_log =
        windowed ? 17 + std::min(6, (level - 10) / 2) : 17;

    std::size_t size = data.size();

    std::vector<std::byte> output;
    std::uint32_t magic = 0xFD2FB528u;
    output.insert(output.end(), reinterpret_cast<std::byte*>(&magic),
                  reinterpret_cast<std::byte*>(&magic) + 4);

    int fcs_size;
    std::uint8_t fcs_code;
    std::uint64_t fcs_value;
    if (size < 256) {
        fcs_code = 0; fcs_size = 1; fcs_value = size;
    } else if (size < 256 + 65536) {
        // A 2-byte Frame_Content_Size stores the value minus 256.
        fcs_code = 1; fcs_size = 2; fcs_value = size - 256;
    } else if (size < (1ULL << 32)) {
        fcs_code = 2; fcs_size = 4; fcs_value = size;
    } else {
        fcs_code = 3; fcs_size = 8; fcs_value = size;
    }
    bool single_seg = (fcs_code == 0);
    std::uint8_t desc = static_cast<std::uint8_t>(
        (1 << 2) | (single_seg ? (1 << 5) : 0) | (fcs_code << 6));
    output.push_back(static_cast<std::byte>(desc));
    if (!single_seg) {
        // Window descriptor: Window_Log = 10 + (byte >> 3); low 3 bits add a
        // fraction of the window (0 here, i.e. an exact power of two).
        output.push_back(
            static_cast<std::byte>((window_log - 10) << 3));
    }
    for (int i = 0; i < fcs_size; ++i) {
        output.push_back(static_cast<std::byte>((fcs_value >> (8 * i)) & 0xFF));
    }

    const auto* p = reinterpret_cast<const std::uint8_t*>(data.data());

    // Split into blocks. The match finder and entropy tables are per block,
    // so tokenization runs in parallel. Emission then happens sequentially so
    // repeat offsets carry from one compressed block to the next (as the
    // decoder requires); raw blocks leave the repeat state untouched.
    struct Block {
        std::size_t off = 0;
        std::size_t len = 0;
        bool last = false;
        std::vector<std::uint8_t> literals;
        std::vector<Seq> seqs;
        std::vector<std::byte> content;  // compressed payload (if any)
        bool compressed = false;
    };
    std::vector<Block> blocks;
    for (std::size_t off = 0; off < size;) {
        std::size_t blen = std::min<std::size_t>(kBlockSize, size - off);
        Block b;
        b.off = off;
        b.len = blen;
        b.last = (off + blen == size);
        blocks.push_back(std::move(b));
        off += blen;
    }

    if (windowed) {
        // Shared match finder across the whole frame: matches may reference
        // any earlier block within the window. Tokenization is sequential
        // because each block extends the shared finder for the next one.
        MatchFinder mf;
        mf.reset(p, size, 1 << window_log);
        for (Block& b : blocks) {
            if (optimal) {
                b.seqs = build_sequences_optimal(mf, b.off, b.off + b.len,
                                                 effort, passes, b.literals);
            } else {
                b.seqs = build_sequences(mf, b.off, b.off + b.len, effort,
                                         lazy, b.literals);
            }
        }
    } else {
        // Independent per-block matching, tokenized in parallel.
        auto tokenize_block = [&](Block& b) {
            MatchFinder mf;
            mf.reset(p + b.off, b.len, kBlockSize);
            if (optimal) {
                b.seqs = build_sequences_optimal(mf, 0, b.len, effort, passes,
                                                 b.literals);
            } else {
                b.seqs = build_sequences(mf, 0, b.len, effort, lazy,
                                         b.literals);
            }
        };

        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 1;
        if (blocks.size() < 2 || hw <= 1) {
            for (Block& b : blocks) {
                tokenize_block(b);
            }
        } else {
            const unsigned nw =
                std::min<unsigned>(hw, static_cast<unsigned>(blocks.size()));
            std::atomic<std::size_t> next{0};
            std::vector<std::thread> workers;
            workers.reserve(nw);
            for (unsigned t = 0; t < nw; ++t) {
                workers.emplace_back([&]() {
                    for (;;) {
                        std::size_t i =
                            next.fetch_add(1, std::memory_order_relaxed);
                        if (i >= blocks.size()) break;
                        tokenize_block(blocks[i]);
                    }
                });
            }
            for (auto& w : workers) w.join();
        }
    }

    // Emit blocks in order, threading the repeat-offset state through the
    // compressed ones.
    RepeatOffsets repeat;
    for (Block& b : blocks) {
        if (!b.seqs.empty()) {
            std::vector<std::byte> content;
            emit_literals(content, b.literals);
            RepeatOffsets trial = repeat;
            emit_sequences(content, b.seqs, trial);
            if (content.size() < b.len) {
                b.content = std::move(content);
                b.compressed = true;
                repeat = trial;
            }
        }

        const std::byte* payload;
        std::size_t clen;
        if (b.compressed) {
            payload = b.content.data();
            clen = b.content.size();
        } else {
            payload = reinterpret_cast<const std::byte*>(p + b.off);
            clen = b.len;
        }
        std::uint32_t type = b.compressed ? 2u : 0u;
        std::uint32_t cs = static_cast<std::uint32_t>(clen);
        std::uint32_t blk_hdr = (b.last ? 1u : 0u) | (type << 1) | (cs << 3);
        output.push_back(static_cast<std::byte>(blk_hdr & 0xFF));
        output.push_back(static_cast<std::byte>((blk_hdr >> 8) & 0xFF));
        output.push_back(static_cast<std::byte>((blk_hdr >> 16) & 0xFF));
        output.insert(output.end(), payload, payload + clen);
    }

    std::uint64_t checksum = xxhash64(data);
    std::uint32_t csum = static_cast<std::uint32_t>(checksum);
    output.insert(output.end(), reinterpret_cast<std::byte*>(&csum),
                  reinterpret_cast<std::byte*>(&csum) + 4);

    if (output.size() >= size) return {};
    return output;
}

}  // namespace fzip::zstd
