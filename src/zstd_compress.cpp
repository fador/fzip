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
#include <bit>
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

// --- LZ77 match finder: dual hash (3-byte and 4-byte) + hash chains over a
// configurable window. Chains persist across blocks so matches can reach
// into earlier blocks (bounded by `window`). ---
constexpr int kHashBits = 16;
constexpr int kHashSize = 1 << kHashBits;
constexpr int kHashBits4 = 17;
constexpr int kHashSize4 = 1 << kHashBits4;
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

inline auto hash4(const std::uint8_t* p) -> std::uint32_t {
    std::uint32_t v = static_cast<std::uint32_t>(p[0]) |
                      (static_cast<std::uint32_t>(p[1]) << 8) |
                      (static_cast<std::uint32_t>(p[2]) << 16) |
                      (static_cast<std::uint32_t>(p[3]) << 24);
    return (v * 2654435761u) >> (32 - kHashBits4);
}

inline auto match_len_fast(const std::uint8_t* a, const std::uint8_t* b,
                           int maxl) -> int {
    int l = 0;
    while (l + 8 <= maxl) {
        std::uint64_t va, vb;
        std::memcpy(&va, a + l, 8);
        std::memcpy(&vb, b + l, 8);
        if (va != vb) {
            return l + (std::countr_zero(va ^ vb) >> 3);
        }
        l += 8;
    }
    while (l < maxl && a[l] == b[l]) ++l;
    return l;
}

constexpr int kMaxZstdCands = 16;

struct MatchFinder {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    int window = kBlockSize;  // power of two
    int mask = kBlockSize - 1;
    std::vector<int> head;   // kHashSize (3-byte)
    std::vector<int> prev;   // window
    std::vector<int> head4;  // kHashSize4 (4-byte)
    std::vector<int> prev4;  // window

    void reset(const std::uint8_t* d, std::size_t n, int w) {
        data = d;
        size = n;
        window = w;
        mask = w - 1;
        head.assign(kHashSize, -1);
        prev.assign(static_cast<std::size_t>(w), -1);
        head4.assign(kHashSize4, -1);
        prev4.assign(static_cast<std::size_t>(w), -1);
    }

    void insert(std::size_t pos) {
        if (pos + kMinMatch > size) return;
        int p = static_cast<int>(pos);
        const std::size_t wpos =
            static_cast<std::size_t>(p) & static_cast<std::size_t>(mask);

        std::uint32_t h3 = hash3(data + pos);
        prev[wpos] = head[h3];
        head[h3] = p;

        if (pos + 4 <= size) {
            std::uint32_t h4 = hash4(data + pos);
            prev4[wpos] = head4[h4];
            head4[h4] = p;
        }
    }

    // Find the best match at `pos` that ends no later than `end`.
    auto find(std::size_t pos, std::size_t end, int effort) const -> Match {
        Match best;
        if (pos + kMinMatch > end) return best;
        const int maxl = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(kMaxMatch), end - pos));
        int nice_len = (effort >= 256) ? maxl : std::min(maxl, std::max(64, effort));
        int limit = static_cast<int>(pos) - window;
        if (limit < 0) limit = 0;

        // 1. Search 4-byte chain first for 4+ byte matches
        if (pos + 4 <= end) {
            std::uint32_t h4 = hash4(data + pos);
            int cand = head4[h4];
            int tries = effort;
            while (cand >= 0 && tries-- > 0) {
                if (cand < limit) break;
                if (best.length >= 4) {
                    if (best.length >= maxl) break;
                    if (data[cand + best.length] != data[pos + best.length] ||
                        data[cand + best.length - 1] !=
                            data[pos + best.length - 1]) {
                        cand = prev4[static_cast<std::size_t>(cand) &
                                     static_cast<std::size_t>(mask)];
                        continue;
                    }
                }
                int l = match_len_fast(data + cand, data + pos, maxl);
                if (l >= 4 && l > best.length) {
                    best.length = l;
                    best.distance = static_cast<int>(pos) - cand;
                    if (l >= nice_len) break;
                }
                cand = prev4[static_cast<std::size_t>(cand) &
                             static_cast<std::size_t>(mask)];
            }
        }

        // 2. Search 3-byte chain for a close 3-byte match if no 4+ byte match
        // was found (or if the 4-byte match is very far away).
        if (best.length < 4 || (best.length == 4 && best.distance > 32768)) {
            std::uint32_t h3 = hash3(data + pos);
            int cand = head[h3];
            int tries = std::min(effort, 16);
            while (cand >= 0 && tries-- > 0) {
                if (cand < limit) break;
                int d = static_cast<int>(pos) - cand;
                if (best.length == 4 && d >= 512) break;
                int l = match_len_fast(data + cand, data + pos, maxl);
                if (l >= kMinMatch) {
                    if (best.length < 4 || l > best.length ||
                        (l == 3 && d < 512 && best.distance > 32768)) {
                        best.length = l;
                        best.distance = d;
                    }
                    break;  // First valid 3-byte match is the closest
                }
                cand = prev[static_cast<std::size_t>(cand) &
                            static_cast<std::size_t>(mask)];
            }
        }
        return best;
    }

    // Collect Pareto-optimal matches at `pos` into `out` (sorted by increasing
    // distance and length).
    auto find_candidates(std::size_t pos, std::size_t end, int effort,
                         Match (&out)[kMaxZstdCands]) const -> int {
        int n = 0;
        if (pos + kMinMatch > end) return 0;
        const int maxl = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(kMaxMatch), end - pos));
        int limit = static_cast<int>(pos) - window;
        if (limit < 0) limit = 0;
        int best_len = 0;

        auto consider = [&](int cand, int min_req) -> bool {
            if (cand < limit) return false;
            const int d = static_cast<int>(pos) - cand;
            if (n > 0 && d >= out[n - 1].distance && best_len >= min_req) {
                if (best_len >= maxl) return false;
                if (data[cand + best_len] != data[pos + best_len]) {
                    return true;
                }
            }
            const int l = match_len_fast(data + cand, data + pos, maxl);
            if (l >= min_req) {
                if (l > best_len) best_len = l;
                bool dominated = false;
                for (int j = 0; j < n; ++j) {
                    if (out[j].distance <= d && out[j].length >= l) {
                        dominated = true;
                        break;
                    }
                }
                if (!dominated) {
                    int w = 0;
                    for (int j = 0; j < n; ++j) {
                        if (!(out[j].distance >= d && out[j].length <= l)) {
                            out[w++] = out[j];
                        }
                    }
                    n = w;
                    if (n < kMaxZstdCands) {
                        int ins = 0;
                        while (ins < n && out[ins].distance < d) ++ins;
                        for (int j = n; j > ins; --j) out[j] = out[j - 1];
                        out[ins] = Match{d, l};
                        ++n;
                    } else if (l > out[n - 1].length) {
                        out[n - 1] = Match{d, l};
                    }
                }
                if (l >= maxl) return false;
            }
            return true;
        };

        // 1. Closest 3-byte match from head3 (all 4+ byte matches are in head4).
        {
            int cand = head[hash3(data + pos)];
            int tries = 8;
            while (cand >= 0 && tries-- > 0) {
                if (cand < limit) break;
                int prev_n = n;
                if (!consider(cand, kMinMatch) || n > prev_n) break;
                cand = prev[static_cast<std::size_t>(cand) &
                            static_cast<std::size_t>(mask)];
            }
        }
        // 2. 4-byte hash chain for all longer Pareto matches.
        if (pos + 4 <= end) {
            int cand = head4[hash4(data + pos)];
            int tries = effort;
            while (cand >= 0 && tries-- > 0) {
                if (!consider(cand, 4)) break;
                cand = prev4[static_cast<std::size_t>(cand) &
                             static_cast<std::size_t>(mask)];
            }
        }
        return n;
    }
};

// One sequence: `lit_len` literals, then a match of `match_len` at
// `match_dist` bytes back.
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

// Shannon cost in bits for a symbol with the given frequency.
inline auto sym_cost(int freq, int total) -> float {
    if (freq <= 0 || total <= 0) return 11.0f;
    return -std::log2(static_cast<float>(freq) / static_cast<float>(total));
}

// --- Code tables (RFC 8878 §4.2.2) ---

inline Code lit_len_code(int len) {
    if (len < 16) {
        return {len, 0, 0};
    }
    static const int kBase[36] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048,
        4096, 8192, 16384, 32768, 65536};
    static const int kExtra[36] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    for (int c = 35; c >= 16; --c) {
        if (len >= kBase[c]) return {c, kExtra[c], len - kBase[c]};
    }
    return {0, 0, 0};
}

inline Code match_len_code(int len) {
    if (len >= 3 && len <= 34) {
        return {len - 3, 0, 0};
    }
    static const int kBase[53] = {
        3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027,
        2051, 4099, 8195, 16387, 32771, 65539};
    static const int kExtra[53] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    for (int c = 52; c >= 32; --c) {
        if (len >= kBase[c]) return {c, kExtra[c], len - kBase[c]};
    }
    return {0, 0, 0};
}

// Offset code: Offset_Value = (1 << code) + extra, offset = Offset_Value - 3.
// Codes 0..1 are repeat codes; explicit offsets use code >= 2.
inline Code offset_code(int dist) {
    int v = dist + 3;
    int code = std::bit_width(static_cast<std::uint32_t>(v)) - 1;
    if (code < 2) code = 2;  // never use repeat codes for explicit offsets
    return {code, code, v - (1 << code)};
}

// Pick the cheapest offset code for `dist` given the current repeat state,
// updating `rep` to exactly the state the decoder will have afterwards.
// Mirrors ZSTD_decodeSequence's repeat-offset logic (RFC 8878 §3.1.1.5).
inline Code choose_offset(int dist, int lit_len, RepeatOffsets& rep) {
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

struct RepMatch {
    int dist = 0;
    int len = 0;
    int rep_idx = -1;  // 0, 1, or 2
};

inline auto find_rep_match(const std::uint8_t* data, std::size_t pos,
                           std::size_t end, const RepeatOffsets& rep,
                           int min_len = 3) -> RepMatch {
    RepMatch best;
    const int maxl = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(kMaxMatch), end - pos));
    if (maxl < min_len) return best;
    for (int idx = 0; idx < 3; ++idx) {
        int d = rep.offsets[idx];
        if (d <= 0 || static_cast<int>(pos) < d) continue;
        const std::size_t cand = pos - static_cast<std::size_t>(d);
        if (data[cand] != data[pos]) continue;
        int l = match_len_fast(data + cand, data + pos, maxl);
        if (l >= min_len && l > best.len) {
            best.dist = d;
            best.len = l;
            best.rep_idx = idx;
        }
    }
    return best;
}

// LZ77 + lazy matching over `[start, end)`. Positions are absolute within
// `mf.data`; the shared finder may reference earlier blocks. Tracks and
// exploits repeat offsets.
auto build_sequences(MatchFinder& mf, std::size_t start, std::size_t end,
                     int effort, bool lazy,
                     std::vector<std::uint8_t>& literals,
                     RepeatOffsets& rep) -> std::vector<Seq> {
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
        RepMatch rep_m = find_rep_match(mf.data, pos, end, rep);
        Match m = mf.find(pos, end, effort);
        if (m.length == 3 && m.distance > 16384) {
            m = {};
        }

        bool use_rep = (rep_m.len >= kMinMatch);
        if (use_rep && m.length > rep_m.len) {
            int extra_cost_bytes = (m.distance > 256) ? 2 : 1;
            if (m.length <= rep_m.len + extra_cost_bytes) {
                use_rep = true;
            } else {
                use_rep = false;
            }
        }
        if (use_rep) {
            m.length = rep_m.len;
            m.distance = rep_m.dist;
        }

        if (m.length < kMinMatch) {
            mf.insert(pos);
            ++pos;
            continue;
        }

        // Lazy matching: insert pos, then look one byte ahead.
        if (lazy && pos + 1 < end) {
            mf.insert(pos);
            RepMatch rep_m2 = find_rep_match(mf.data, pos + 1, end, rep);
            Match m2 = mf.find(pos + 1, end, effort);
            if (m2.length == 3 && m2.distance > 16384) {
                m2 = {};
            }
            bool use_rep2 = (rep_m2.len >= kMinMatch);
            if (use_rep2 && m2.length > rep_m2.len) {
                int extra_cost_bytes = (m2.distance > 256) ? 2 : 1;
                if (m2.length <= rep_m2.len + extra_cost_bytes) {
                    use_rep2 = true;
                } else {
                    use_rep2 = false;
                }
            }
            if (use_rep2) {
                m2.length = rep_m2.len;
                m2.distance = rep_m2.dist;
            }

            if (m2.length > m.length) {
                ++pos;
                m = m2;
                flush_literals(lit_start, pos);
                Seq s;
                s.lit_len = static_cast<int>(pos - lit_start);
                s.match_len = m.length;
                s.match_dist = m.distance;
                choose_offset(s.match_dist, s.lit_len, rep);
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
        choose_offset(s.match_dist, s.lit_len, rep);
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

// Forward-DP optimal (shortest-path) LZ77 parse over `[start, end)`.
// Collects Pareto-optimal match candidates per position in a single forward
// pass, then runs `passes` iterations of forward DP that dynamically track the
// 3 repeat offsets (`rep[i]`) and active literal run (`lit_run[i]`) at every
// position in the block, pricing literal lengths, match lengths, explicit
// offsets, and repeat offsets (`RO1`, `RO2`, `RO3`, and `RO1-1` when ll==0).
auto build_sequences_optimal(MatchFinder& mf, std::size_t start,
                             std::size_t end, int effort, int passes,
                             std::vector<std::uint8_t>& literals,
                             RepeatOffsets& rep)
    -> std::vector<Seq> {
    const int N = static_cast<int>(end - start);
    if (N <= 0) {
        literals.clear();
        return {};
    }

    const RepeatOffsets initial_rep = rep;
    const int window_limit = mf.window;

    // 1. Forward pass: collect Pareto-optimal matches at every position.
    std::vector<Match> pool;
    pool.reserve(static_cast<std::size_t>(N) * 2);
    std::vector<int> cand_start(static_cast<std::size_t>(N) + 1, 0);
    for (int p = 0; p < N; ++p) {
        cand_start[static_cast<std::size_t>(p)] = static_cast<int>(pool.size());
        Match local[kMaxZstdCands];
        const std::size_t pos = start + static_cast<std::size_t>(p);
        const int nc = mf.find_candidates(pos, end, effort, local);
        for (int j = 0; j < nc; ++j) pool.push_back(local[j]);
        mf.insert(pos);
    }
    cand_start[static_cast<std::size_t>(N)] = static_cast<int>(pool.size());

    // Max match length representable by each match-length code.
    int ml_range_max[53];
    for (int c = 0; c < 53; ++c) {
        ml_range_max[c] = matchlen_code_to_base(c) +
                          (1 << matchlen_code_to_extra(c)) - 1;
    }

    // 2. Seed symbol statistics with a repeat-aware greedy parse over `pool`.
    int lit_freq[256] = {0};
    int ll_freq[36] = {0};
    int of_freq[32] = {0};
    int ml_freq[53] = {0};
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
            RepMatch rm = find_rep_match(mf.data, pos, end, greedy_rep);
            if (rm.len >= kMinMatch && rm.dist <= window_limit && rm.len + 2 >= L) {
                L = rm.len;
                d = rm.dist;
            }
            if (L >= kMinMatch) {
                ll_freq[lit_len_code(lit_run).code]++;
                Code oc = choose_offset(d, lit_run, greedy_rep);
                of_freq[oc.code]++;
                ml_freq[match_len_code(L).code]++;
                ++n_seq;
                p += L;
                lit_run = 0;
            } else {
                lit_freq[mf.data[pos]]++;
                ++n_lit;
                ++p;
                ++lit_run;
            }
        }
        // Cold-start protection so all candidate codes have non-zero priors.
        for (const auto& c : pool) {
            int mc = match_len_code(c.length).code;
            if (ml_freq[mc] == 0) { ml_freq[mc] = 1; ++n_seq; }
            int oc = offset_code(c.distance).code;
            if (oc < 32 && of_freq[oc] == 0) { of_freq[oc] = 1; }
        }
        if (of_freq[0] == 0) of_freq[0] = 1;
        if (of_freq[1] == 0) of_freq[1] = 1;
    }

    constexpr float kInf = 1e30f;
    std::vector<float> cost(static_cast<std::size_t>(N) + 1, kInf);
    std::vector<int> lit_run_at(static_cast<std::size_t>(N) + 1, 0);
    std::vector<RepeatOffsets> rep_at(static_cast<std::size_t>(N) + 1, initial_rep);
    std::vector<int> back_len(static_cast<std::size_t>(N) + 1, 1);
    std::vector<int> back_dist(static_cast<std::size_t>(N) + 1, 0);

    std::vector<Seq> best_seqs;
    std::vector<std::uint8_t> best_literals;
    float best_total_cost = kInf;

    if (passes < 1) passes = 1;
    for (int pass = 0; pass < passes; ++pass) {
        float lit_cost[256];
        for (int s = 0; s < 256; ++s) {
            float c = sym_cost(lit_freq[s], n_lit);
            if (c > 11.0f) c = 11.0f;
            if (c < 1.0f) c = 1.0f;
            lit_cost[s] = c;
        }
        float llc_cost[36];
        for (int c = 0; c < 36; ++c) {
            llc_cost[c] = sym_cost(ll_freq[c], n_seq) +
                          static_cast<float>(litlen_code_to_extra(c));
        }
        float mlc_cost[53];
        for (int c = 0; c < 53; ++c) {
            mlc_cost[c] = sym_cost(ml_freq[c], n_seq) +
                          static_cast<float>(matchlen_code_to_extra(c));
        }
        float ofc_cost[32];
        for (int c = 0; c < 32; ++c) {
            int extra = (c == 0) ? 0 : (c == 1 ? 1 : c);
            ofc_cost[c] = sym_cost(of_freq[c], n_seq) + static_cast<float>(extra);
        }

        // Precompute incremental literal-length cost delta: when emitting a
        // sequence after `ll` literals, the sequence header pays `llc_cost[code(ll)]`.
        auto ll_bits = [&](int ll) -> float {
            return llc_cost[lit_len_code(ll).code];
        };

        std::fill(cost.begin(), cost.end(), kInf);
        cost[0] = 0.0f;
        lit_run_at[0] = 0;
        rep_at[0] = initial_rep;

        for (int i = 0; i < N; ++i) {
            const float cur_cost = cost[static_cast<std::size_t>(i)];
            if (cur_cost >= kInf) continue;

            const std::size_t pos = start + static_cast<std::size_t>(i);
            const int ll = lit_run_at[static_cast<std::size_t>(i)];
            const RepeatOffsets cur_rep = rep_at[static_cast<std::size_t>(i)];

            // 1. Literal transition: i -> i + 1
            {
                const float nc = cur_cost + lit_cost[mf.data[pos]];
                const std::size_t next_i = static_cast<std::size_t>(i + 1);
                if (nc < cost[next_i]) {
                    cost[next_i] = nc;
                    lit_run_at[next_i] = ll + 1;
                    rep_at[next_i] = cur_rep;
                    back_len[next_i] = 1;
                    back_dist[next_i] = 0;
                }
            }

            const float base_seq_cost = cur_cost + ll_bits(ll);
            const int maxl = std::min(kMaxMatch, N - i);
            if (maxl < kMinMatch) continue;

            auto try_match = [&](int d, int max_match_len, float dc,
                                 const RepeatOffsets& next_rep, int min_l) {
                const int dense_limit = std::min(max_match_len, 48);
                for (int L = min_l; L <= dense_limit; ++L) {
                    const float mc = base_seq_cost + dc +
                                     mlc_cost[match_len_code(L).code];
                    const std::size_t dst = static_cast<std::size_t>(i + L);
                    if (mc < cost[dst]) {
                        cost[dst] = mc;
                        lit_run_at[dst] = 0;
                        rep_at[dst] = next_rep;
                        back_len[dst] = L;
                        back_dist[dst] = d;
                    }
                }
                if (max_match_len > dense_limit) {
                    const int c_start = match_len_code(dense_limit + 1).code;
                    const int c_end = match_len_code(max_match_len).code;
                    for (int c = c_start; c <= c_end; ++c) {
                        int L = std::min(ml_range_max[c], max_match_len);
                        if (L <= dense_limit) continue;
                        const float mc = base_seq_cost + dc + mlc_cost[c];
                        const std::size_t dst = static_cast<std::size_t>(i + L);
                        if (mc < cost[dst]) {
                            cost[dst] = mc;
                            lit_run_at[dst] = 0;
                            rep_at[dst] = next_rep;
                            back_len[dst] = L;
                            back_dist[dst] = d;
                        }
                    }
                }
            };

            // 2. Dynamic repeat-offset matches from `cur_rep`.
            {
                int rep_dists[3];
                if (ll > 0) {
                    rep_dists[0] = cur_rep.offsets[0];
                    rep_dists[1] = cur_rep.offsets[1];
                    rep_dists[2] = cur_rep.offsets[2];
                } else {
                    rep_dists[0] = cur_rep.offsets[1];
                    rep_dists[1] = cur_rep.offsets[2];
                    rep_dists[2] = (cur_rep.offsets[0] > 1)
                                       ? (cur_rep.offsets[0] - 1)
                                       : 0;
                }
                for (int r_idx = 0; r_idx < 3; ++r_idx) {
                    const int rd = rep_dists[r_idx];
                    if (rd <= 0 || rd > window_limit ||
                        static_cast<int>(pos) < rd) {
                        continue;
                    }
                    const std::size_t cand = pos - static_cast<std::size_t>(rd);
                    if (mf.data[cand] != mf.data[pos]) continue;
                    const int rl =
                        match_len_fast(mf.data + cand, mf.data + pos, maxl);
                    if (rl >= kMinMatch) {
                        RepeatOffsets next_rep = cur_rep;
                        Code oc = choose_offset(rd, ll, next_rep);
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
                    RepeatOffsets next_rep = cur_rep;
                    Code oc = choose_offset(c.distance, ll, next_rep);
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
            const int L = back_len[static_cast<std::size_t>(p)];
            const int d = back_dist[static_cast<std::size_t>(p)];
            if (L <= 1) {
                --p;
            } else {
                const int match_start = p - L;
                const int lrun = lit_run_at[static_cast<std::size_t>(match_start)];
                Seq s;
                s.lit_len = lrun;
                s.match_len = L;
                s.match_dist = d;
                seqs.push_back(s);
                p = match_start - lrun;
            }
        }
        std::reverse(seqs.begin(), seqs.end());

        // Rebuild literal stream for this parse.
        std::vector<std::uint8_t> pass_lits;
        pass_lits.reserve(static_cast<std::size_t>(N));
        std::size_t cur_pos = start;
        for (const Seq& s : seqs) {
            for (int i = 0; i < s.lit_len; ++i) {
                pass_lits.push_back(mf.data[cur_pos + static_cast<std::size_t>(i)]);
            }
            cur_pos += static_cast<std::size_t>(s.lit_len + s.match_len);
        }
        while (cur_pos < end) {
            pass_lits.push_back(mf.data[cur_pos++]);
        }

        if (cost[static_cast<std::size_t>(N)] < best_total_cost ||
            best_seqs.empty()) {
            best_total_cost = cost[static_cast<std::size_t>(N)];
            best_seqs = seqs;
            best_literals = pass_lits;
        }

        // Recompute statistics from this parse for the next iteration.
        if (pass + 1 < passes) {
            std::fill_n(lit_freq, 256, 0);
            std::fill_n(ll_freq, 36, 0);
            std::fill_n(of_freq, 32, 0);
            std::fill_n(ml_freq, 53, 0);
            for (std::uint8_t b : pass_lits) lit_freq[b]++;
            n_lit = static_cast<int>(pass_lits.size());
            RepeatOffsets pass_rep = initial_rep;
            for (const Seq& s : seqs) {
                ll_freq[lit_len_code(s.lit_len).code]++;
                Code oc = choose_offset(s.match_dist, s.lit_len, pass_rep);
                of_freq[oc.code]++;
                ml_freq[match_len_code(s.match_len).code]++;
            }
            n_seq = static_cast<int>(seqs.size());
        }
    }

    literals = std::move(best_literals);
    RepeatOffsets end_rep = initial_rep;
    for (const Seq& s : best_seqs) {
        choose_offset(s.match_dist, s.lit_len, end_rep);
    }
    rep = end_rep;

    return best_seqs;
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

// Size-field width (k) for a compressed/treeless literals section.
auto literals_k(int regen, int csize) -> int {
    const int m = std::max(regen, csize);
    if (m <= 1023) return 10;
    if (m <= 16383) return 14;
    return 18;
}

// Emit the literals section, choosing Huffman (Compressed, 1-stream or 4-stream),
// reused-table Huffman (Treeless, 1-stream or 4-stream), RLE, or Raw — whichever
// is smallest. `prev_t` carries the last Huffman table across blocks.
void emit_literals(std::vector<std::byte>& out,
                   const std::vector<std::uint8_t>& lits, HufTable* prev_t) {
    const int n = static_cast<int>(lits.size());

    // Try Huffman-coded literals (1-stream when n <= 1023, and 4-stream) with
    // a fresh table (Compressed) and, when available, reusing the previous
    // table (Treeless).
    if (n >= 18) {
        HufTable new_t;
        const bool have_table = huf_literals_table(lits.data(), n, new_t);
        const bool can_reuse = (prev_t != nullptr && prev_t->table_log != 0);

        int best_type = 0;
        int best_sf = 0;
        int best_k = 0;
        std::vector<std::byte> best_body;
        std::size_t best_total = static_cast<std::size_t>(n);

        auto try_candidate = [&](int type, bool single_stream,
                                 const HufTable& tbl) {
            if (single_stream && n > 1023) return;
            std::vector<std::byte> body;
            bool ok = (type == 2)
                          ? huf_encode_with_table(tbl, lits.data(), n, body,
                                                  single_stream)
                          : huf_encode_streams(tbl, lits.data(), n, body,
                                               single_stream);
            if (!ok) return;
            if (single_stream && body.size() > 1023) return;
            const int k = single_stream
                              ? 10
                              : literals_k(n, static_cast<int>(body.size()));
            const int sf =
                single_stream ? 0 : ((k == 10) ? 1 : (k == 14) ? 2 : 3);
            const std::size_t hdr_bytes =
                static_cast<std::size_t>((4 + 2 * k) / 8);
            const std::size_t total = hdr_bytes + body.size();
            if (total < best_total) {
                best_total = total;
                best_type = type;
                best_sf = sf;
                best_k = k;
                best_body = std::move(body);
            }
        };

        if (have_table) {
            if (n <= 1023) try_candidate(2, true, new_t);
            try_candidate(2, false, new_t);
        }
        if (can_reuse) {
            if (n <= 1023) try_candidate(3, true, *prev_t);
            try_candidate(3, false, *prev_t);
        }

        if (best_type != 0) {
            put_literals_header(out, best_type, best_sf, n,
                                static_cast<int>(best_body.size()), best_k);
            out.insert(out.end(), best_body.begin(), best_body.end());
            if (best_type == 2 && prev_t != nullptr) *prev_t = new_t;
            return;
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
    int mode = 0;  // 0 predefined, 1 RLE, 2 FSE, 3 repeat
    int rle_sym = 0;
    FseCTable ct;
    std::vector<std::byte> desc;  // NCount table description (mode 2)
};

// Exact number of FSE bits the encoder will emit for `syms` with `ct`,
// replicating emit_sequences' order (init with the last symbol, encode
// n-2..0, then flush the final state). Extra bits are table-independent.
auto fse_encoded_bits(const FseCTable& ct, const int* syms, int n)
    -> long long {
    if (n <= 0) return 0;
    if (n == 1) return ct.table_log;
    std::uint32_t value = fse_init_cstate2(ct, syms[n - 1]);
    long long bits = 0;
    for (int i = n - 2; i >= 0; --i) {
        const std::uint32_t d = ct.delta_nb_bits[syms[i]];
        const std::uint32_t nb = (value + d) >> 16;
        bits += nb;
        const int idx =
            static_cast<int>(value >> nb) + ct.delta_find_state[syms[i]];
        value = ct.state_table[static_cast<std::size_t>(idx)];
    }
    bits += ct.table_log;  // final state
    return bits;
}

// Alternative FSE normalization that assigns -1 ("less than 1" probability) to
// rare symbols, freeing table states for frequent symbols and shrinking NCount.
void fse_normalize_lowprob(const unsigned* freqs, int total, int max_symbol,
                           int table_log, std::vector<int>& norm) {
    const int table_size = 1 << table_log;
    norm.assign(static_cast<std::size_t>(max_symbol) + 1, 0);
    if (total <= 0) return;
    const long long low_thresh = std::max(1, total >> (table_log + 1));
    long long sum = 0;
    for (int s = 0; s <= max_symbol; ++s) {
        if (freqs[s] == 0) continue;
        if (static_cast<long long>(freqs[s]) <= low_thresh) {
            norm[static_cast<std::size_t>(s)] = -1;
            sum += 1;
        } else {
            long long v =
                (static_cast<long long>(freqs[s]) * table_size + (total / 2)) /
                total;
            if (v < 1) v = 1;
            norm[static_cast<std::size_t>(s)] = static_cast<int>(v);
            sum += v;
        }
    }
    while (sum > table_size) {
        int best = -1;
        for (int s = 0; s <= max_symbol; ++s) {
            if (norm[static_cast<std::size_t>(s)] > 1 &&
                (best < 0 || norm[static_cast<std::size_t>(s)] >
                                 norm[static_cast<std::size_t>(best)])) {
                best = s;
            }
        }
        if (best < 0) break;
        norm[static_cast<std::size_t>(best)]--;
        sum--;
    }
    while (sum < table_size) {
        int best = -1;
        for (int s = 0; s <= max_symbol; ++s) {
            if (freqs[s] > 0 &&
                (best < 0 || freqs[s] > freqs[best])) {
                best = s;
            }
        }
        if (best < 0) break;
        if (norm[static_cast<std::size_t>(best)] == -1) {
            norm[static_cast<std::size_t>(best)] = 2;
        } else {
            norm[static_cast<std::size_t>(best)]++;
        }
        sum++;
    }
}

auto fse_ctable_supports(const FseCTable& ct, const unsigned* counts,
                         int max_present) -> bool {
    if (ct.table_log <= 0) return false;
    if (static_cast<std::size_t>(max_present) >= ct.delta_nb_bits.size()) {
        return false;
    }
    const auto max_nb = static_cast<std::uint32_t>(ct.table_log);
    for (int c = 0; c <= max_present; ++c) {
        if (counts[c] > 0 && (ct.delta_nb_bits[c] >> 16) >= max_nb) {
            return false;
        }
    }
    return true;
}

// Build a sequence FSE stream. Compares exact bit costs across RLE (mode 1),
// Predefined (mode 0), Repeat (mode 3), and inline FSE (mode 2, across all
// valid accuracy logs and standard/low-probability normalizations).
auto make_seq_stream(const int* syms, int n, int max_code, int max_log,
                     const FseCTable& predef, FseCTable* prev_ct) -> SeqStream {
    SeqStream s;
    std::vector<unsigned> counts(static_cast<std::size_t>(max_code) + 1, 0);
    for (int i = 0; i < n; ++i) {
        counts[static_cast<std::size_t>(syms[i])]++;
    }
    int present = 0;
    int max_present = 0;
    for (int c = 0; c <= max_code; ++c) {
        if (counts[static_cast<std::size_t>(c)] > 0) {
            ++present;
            max_present = c;
        }
    }
    if (present <= 1) {
        s.mode = 1;  // RLE
        s.rle_sym = max_present;
        if (prev_ct != nullptr) prev_ct->table_log = 0;
        return s;
    }

    long long best_bits = (1LL << 62);

    // Candidate 1: Mode 0 (Predefined_Mode, 0 header bits).
    if (fse_ctable_supports(predef, counts.data(), max_present)) {
        best_bits = fse_encoded_bits(predef, syms, n);
        s.mode = 0;
        s.ct = predef;
    }

    // Candidate 2: Mode 3 (Repeat_Mode, 0 header bits).
    if (prev_ct != nullptr &&
        fse_ctable_supports(*prev_ct, counts.data(), max_present)) {
        const long long rep_bits = fse_encoded_bits(*prev_ct, syms, n);
        if (rep_bits < best_bits) {
            best_bits = rep_bits;
            s.mode = 3;
            s.ct = *prev_ct;
        }
    }

    // Candidate 3: Mode 2 (FSE_Compressed_Mode, NCount + stream bits).
    if (n >= 16) {
        int min_log = 5;
        while ((1 << min_log) < present) ++min_log;
        if (min_log > max_log) min_log = max_log;

        auto try_norm = [&](const std::vector<int>& norm, int log) {
            int sum = 0;
            for (int c = 0; c <= max_present; ++c) {
                int v = norm[static_cast<std::size_t>(c)];
                if (counts[static_cast<std::size_t>(c)] > 0 && v == 0) return;
                sum += (v == -1) ? 1 : v;
            }
            if (sum != (1 << log)) return;

            std::vector<std::int16_t> norm16(
                static_cast<std::size_t>(max_present) + 1);
            for (int c = 0; c <= max_present; ++c) {
                norm16[static_cast<std::size_t>(c)] =
                    static_cast<std::int16_t>(norm[static_cast<std::size_t>(c)]);
            }
            FseCTable ct = build_fse_ctable(log, norm16.data(), max_present);
            std::vector<std::byte> desc;
            fse_write_ncount(desc, norm.data(), max_present, log);
            const long long bits =
                static_cast<long long>(desc.size()) * 8 +
                fse_encoded_bits(ct, syms, n);
            if (bits < best_bits) {
                best_bits = bits;
                s.mode = 2;
                s.ct = std::move(ct);
                s.desc = std::move(desc);
            }
        };

        for (int log = min_log; log <= max_log; ++log) {
            std::vector<int> norm;
            fse_normalize(counts.data(), n, max_present, log, norm);
            try_norm(norm, log);

            std::vector<int> norm_lp;
            fse_normalize_lowprob(counts.data(), n, max_present, log, norm_lp);
            if (norm_lp != norm) {
                try_norm(norm_lp, log);
            }
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

void emit_sequences(std::vector<std::byte>& out, const std::vector<Seq>& seqs,
                    RepeatOffsets& rep, FseCTable* prev_ll, FseCTable* prev_of,
                    FseCTable* prev_ml) {
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
    SeqStream ll = make_seq_stream(llc.data(), n, 35, 9, ll_ctable(), prev_ll);
    SeqStream of = make_seq_stream(ofc.data(), n, 31, 8, of_ctable(), prev_of);
    SeqStream ml = make_seq_stream(mlc.data(), n, 52, 9, ml_ctable(), prev_ml);

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
    // Single_Segment drops the window descriptor byte; the decoder then uses
    // the content size as its window. Only use it when the content fits the
    // window we actually matched with, so the frame never demands more
    // decoder memory than a windowed frame would (libzstd caps the window).
    // A 1-byte FCS field is only valid in single-segment mode.
    bool single_seg =
        (fcs_code == 0) || size <= (std::size_t{1} << window_log);
    // No content checksum: inside ZIP the entry's CRC-32 already verifies
    // the decompressed data, so the 4-byte XXH64 trailer is redundant.
    std::uint8_t desc = static_cast<std::uint8_t>(
        (single_seg ? (1 << 5) : 0) | (fcs_code << 6));
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
        RepeatOffsets rep;
        for (Block& b : blocks) {
            if (optimal) {
                b.seqs = build_sequences_optimal(mf, b.off, b.off + b.len,
                                                 effort, passes, b.literals,
                                                 rep);
            } else {
                b.seqs = build_sequences(mf, b.off, b.off + b.len, effort,
                                         lazy, b.literals, rep);
            }
        }
    } else {
        // Independent per-block matching, tokenized in parallel.
        auto tokenize_block = [&](Block& b) {
            MatchFinder mf;
            mf.reset(p + b.off, b.len, kBlockSize);
            RepeatOffsets rep;
            if (optimal) {
                b.seqs = build_sequences_optimal(mf, 0, b.len, effort, passes,
                                                 b.literals, rep);
            } else {
                b.seqs = build_sequences(mf, 0, b.len, effort, lazy,
                                         b.literals, rep);
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

    // Emit blocks in order, threading the repeat-offset state, the literal
    // Huffman table, and the sequence FSE tables through the compressed ones.
    RepeatOffsets repeat;
    HufTable prev_huf;
    FseCTable prev_ll, prev_of, prev_ml;
    for (Block& b : blocks) {
        if (!b.seqs.empty()) {
            std::vector<std::byte> content;
            HufTable trial_huf = prev_huf;
            emit_literals(content, b.literals, &trial_huf);
            RepeatOffsets trial = repeat;
            FseCTable trial_ll = prev_ll;
            FseCTable trial_of = prev_of;
            FseCTable trial_ml = prev_ml;
            emit_sequences(content, b.seqs, trial, &trial_ll, &trial_of,
                           &trial_ml);
            if (content.size() < b.len) {
                b.content = std::move(content);
                b.compressed = true;
                repeat = trial;
                prev_huf = std::move(trial_huf);
                prev_ll = std::move(trial_ll);
                prev_of = std::move(trial_of);
                prev_ml = std::move(trial_ml);
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

    if (output.size() >= size) return {};
    return output;
}

}  // namespace fzip::zstd
