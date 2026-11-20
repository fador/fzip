// fzip — Dynamic Universal Word & Structural Token Replacement Transform (WRT).
#include "fzac_wrt.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fzip::fzac_v3 {

namespace {

constexpr std::uint8_t kEscRaw   = 0x80u;
constexpr std::uint8_t kCapFirst = 0x81u;
constexpr std::uint8_t kCapAll   = 0x82u;

constexpr std::uint8_t kTier1Base = 0x83u;  // 64 1-byte codes: 0x83..0xC2
constexpr std::size_t  kTier1Cap  = 64u;

constexpr std::uint8_t kTier2Base = 0xC3u;  // 56 * 128 = 7168 2-byte codes: 0xC3..0xFA
constexpr std::size_t  kTier2Leads = 56u;
constexpr std::size_t  kTier2Cap  = kTier2Leads * 128u;

constexpr std::uint8_t kTier3Base = 0xFBu;  // 4 * 128 * 128 = 65536 3-byte codes: 0xFB..0xFE
constexpr std::size_t  kTier3Leads = 4u;
constexpr std::size_t  kTier3Cap  = kTier3Leads * 128u * 128u;

constexpr std::size_t  kMaxVocab  = kTier1Cap + kTier2Cap + kTier3Cap;
constexpr std::size_t  kMaxTokLen = 40u;

inline auto is_ascii_alpha(std::uint8_t b) -> bool {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z');
}

inline auto is_word_token(std::string_view sv) -> bool {
    if (sv.empty()) return false;
    for (char c : sv) {
        if (!is_ascii_alpha(static_cast<std::uint8_t>(c))) return false;
    }
    return true;
}

inline auto to_upper_ascii(std::uint8_t b) -> std::uint8_t {
    return (b >= 'a' && b <= 'z') ? static_cast<std::uint8_t>(b - ('a' - 'A')) : b;
}

inline void append_varint(std::vector<std::byte>& out, std::uint64_t v) {
    while (v >= 0x80u) {
        out.push_back(static_cast<std::byte>((v & 0x7Fu) | 0x80u));
        v >>= 7;
    }
    out.push_back(static_cast<std::byte>(v));
}

inline auto read_varint(const std::uint8_t* in, std::size_t size,
                        std::size_t& off, const char* err_msg) -> std::uint64_t {
    std::uint64_t val = 0;
    int shift = 0;
    for (;;) {
        if (off >= size || shift > 63) {
            throw std::runtime_error(err_msg);
        }
        const std::uint8_t b = in[off++];
        val |= static_cast<std::uint64_t>(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0u) return val;
        shift += 7;
    }
}

// Classify an alphabetic word run [p, p+len) into:
//   0 = all lower, 1 = FirstCap + lower, 2 = ALLCAPS, 3 = mixed
inline auto classify_word(const std::uint8_t* p, std::size_t len,
                          std::string& lower_out) -> int {
    lower_out.resize(len);
    bool all_lower = true;
    bool all_upper = true;
    bool rest_lower = true;
    for (std::size_t i = 0; i < len; ++i) {
        const std::uint8_t b = p[i];
        const bool up = (b >= 'A' && b <= 'Z');
        if (up) {
            all_lower = false;
            if (i > 0u) rest_lower = false;
            lower_out[i] = static_cast<char>(b + ('a' - 'A'));
        } else {
            all_upper = false;
            lower_out[i] = static_cast<char>(b);
        }
    }
    if (all_lower) return 0;
    if (all_upper) return 2;
    if ((p[0] >= 'A' && p[0] <= 'Z') && rest_lower) return 1;
    return 3;
}

// Try to recognize a structural/markup token starting at `raw[i]`.
// Returns its length (0 if none). Only matches pure ASCII tokens in [0x09..0x7E].
inline auto match_struct_span(const std::uint8_t* raw, std::size_t n,
                              std::size_t i) -> std::size_t {
    const std::uint8_t b = raw[i];

    // 1. XML/HTML tag (and chained whitespace+tag spans like "</text>\n    </revision>")
    if (b == '<' && i + 2u < n) {
        const std::uint8_t b1 = raw[i + 1u];
        if (b1 == '/' || is_ascii_alpha(b1) || b1 == '!' || b1 == '?') {
            std::size_t j = i + 2u;
            const std::size_t lim = std::min(n, i + kMaxTokLen);
            std::size_t best_end = 0u;
            while (j < lim) {
                const std::uint8_t c = raw[j];
                if (c == '>') {
                    best_end = j + 1u;
                    // Check if followed by \n + spaces + another <tag> within kMaxTokLen
                    std::size_t k = best_end;
                    if (k < lim && raw[k] == '\n') {
                        ++k;
                        while (k < lim && (raw[k] == ' ' || raw[k] == '\t')) ++k;
                        if (k + 2u < lim && raw[k] == '<' &&
                            (raw[k + 1u] == '/' || is_ascii_alpha(raw[k + 1u]))) {
                            j = k + 2u;
                            continue;
                        }
                    }
                    return best_end - i;
                }
                if (c < 0x20u || c >= 0x7Fu || c == '<') break;
                ++j;
            }
            if (best_end > i) return best_end - i;
        }
    }

    // 2. HTML/XML entity: &amp; &quot; &lt; &gt; &nbsp; etc.
    if (b == '&' && i + 3u < n && is_ascii_alpha(raw[i + 1u])) {
        std::size_t j = i + 2u;
        const std::size_t lim = std::min(n, i + 12u);
        while (j < lim) {
            const std::uint8_t c = raw[j];
            if (c == ';') return (j - i) + 1u;
            if (!is_ascii_alpha(c)) break;
            ++j;
        }
    }

    // 3. Newline + leading spaces/tabs (indentation lines!)
    if (b == '\n' && i + 2u <= n && (raw[i + 1u] == ' ' || raw[i + 1u] == '\t')) {
        std::size_t j = i + 1u;
        const std::size_t lim = std::min(n, i + 24u);
        while (j < lim && (raw[j] == ' ' || raw[j] == '\t')) ++j;
        return j - i;
    }

    // 4. Repeated punctuation / whitespace / operators (e.g. "[[", "]]", "==", "    ", "::", "->")
    if (b == ' ' && i + 2u <= n && raw[i + 1u] == ' ') {
        std::size_t j = i + 2u;
        const std::size_t lim = std::min(n, i + 16u);
        while (j < lim && raw[j] == ' ') ++j;
        return j - i;
    }

    if ((b == '[' || b == ']' || b == '{' || b == '}' || b == '=' ||
         b == ':' || b == '-' || b == '/' || b == '*' || b == '\'') &&
        i + 2u <= n) {
        std::size_t j = i + 1u;
        const std::size_t lim = std::min(n, i + 6u);
        while (j < lim && raw[j] == b) ++j;
        if (j - i >= 2u) return j - i;
        // Common 2-char operators: ->, !=, <=, >=, ==, ::, /*, */
        const std::uint8_t b1 = raw[i + 1u];
        if ((b == '-' && b1 == '>') || (b == '/' && b1 == '*') || (b == '*' && b1 == '/')) {
            return 2u;
        }
    }

    return 0u;
}

inline void emit_code(std::vector<std::uint8_t>& out, std::uint32_t id) {
    if (id < kTier1Cap) {
        out.push_back(static_cast<std::uint8_t>(kTier1Base + id));
    } else if (id < kTier1Cap + kTier2Cap) {
        const std::uint32_t v = id - static_cast<std::uint32_t>(kTier1Cap);
        out.push_back(static_cast<std::uint8_t>(kTier2Base + (v >> 7)));
        out.push_back(static_cast<std::uint8_t>(0x80u | (v & 0x7Fu)));
    } else {
        const std::uint32_t v = id - static_cast<std::uint32_t>(kTier1Cap + kTier2Cap);
        out.push_back(static_cast<std::uint8_t>(kTier3Base + (v >> 14)));
        out.push_back(static_cast<std::uint8_t>(0x80u | ((v >> 7) & 0x7Fu)));
        out.push_back(static_cast<std::uint8_t>(0x80u | (v & 0x7Fu)));
    }
}

void serialize_tier(std::vector<std::byte>& out,
                    const std::vector<std::string>& words) {
    append_varint(out, words.size());
    std::string prev;
    for (const auto& w : words) {
        std::size_t lcp = 0u;
        const std::size_t max_lcp = std::min<std::size_t>({prev.size(), w.size(), 255u});
        while (lcp < max_lcp && prev[lcp] == w[lcp]) ++lcp;
        const std::size_t suf = w.size() - lcp;
        out.push_back(static_cast<std::byte>(lcp));
        out.push_back(static_cast<std::byte>(suf));
        for (std::size_t k = lcp; k < w.size(); ++k) {
            out.push_back(static_cast<std::byte>(w[k]));
        }
        prev = w;
    }
}

void deserialize_tier(const std::uint8_t* in, std::size_t size,
                      std::size_t& off, std::size_t cap,
                      std::vector<std::string>& vocab_out) {
    const std::uint64_t cnt64 =
        read_varint(in, size, off, "truncated FZAC WRT dictionary count");
    if (cnt64 > cap) {
        throw std::runtime_error("invalid FZAC WRT dictionary tier count");
    }
    const std::size_t cnt = static_cast<std::size_t>(cnt64);
    std::string prev;
    for (std::size_t i = 0; i < cnt; ++i) {
        if (off + 2u > size) {
            throw std::runtime_error("truncated FZAC WRT dictionary entry");
        }
        const std::size_t lcp = in[off++];
        const std::size_t suf = in[off++];
        if (lcp > prev.size() || suf > size - off || lcp + suf == 0u || lcp + suf > kMaxTokLen) {
            throw std::runtime_error("corrupted FZAC WRT dictionary entry");
        }
        std::string w;
        w.reserve(lcp + suf);
        w.append(prev, 0u, lcp);
        w.append(reinterpret_cast<const char*>(in + off), suf);
        off += suf;
        prev = w;
        vocab_out.push_back(std::move(w));
    }
}

struct SvHash {
    using is_transparent = void;
    auto operator()(std::string_view sv) const noexcept -> std::size_t {
        std::uint64_t h = 0xCBF29CE484222325ULL;
        for (unsigned char c : sv) {
            h ^= c;
            h *= 0x100000001B3ULL;
        }
        return static_cast<std::size_t>(h);
    }
};

struct SvEq {
    using is_transparent = void;
    auto operator()(std::string_view a, std::string_view b) const noexcept -> bool {
        return a == b;
    }
};

}  // namespace

auto wrt_forward(const std::uint8_t* raw, std::size_t n) -> WrtEncoded {
    WrtEncoded res{};
    if (n < 16384u) return res;

    // Quick text-suitability check across evenly spaced samples.
    std::size_t printable = 0u;
    std::size_t high_bytes = 0u;
    const std::size_t check_n = std::min<std::size_t>(n, 32768u);
    const std::size_t step = std::max<std::size_t>(1u, n / check_n);
    std::size_t checked = 0u;
    for (std::size_t i = 0; i < n && checked < check_n; i += step, ++checked) {
        const std::uint8_t b = raw[i];
        if ((b >= 32u && b <= 126u) || b == '\n' || b == '\r' || b == '\t') {
            ++printable;
        } else if (b >= 0x80u) {
            ++high_bytes;
        }
    }
    if (checked == 0u || printable * 100u < checked * 82u || high_bytes * 100u > checked * 10u) {
        return res;
    }

    // Pass 1: Discover frequent alphabetic words and structural tokens.
    // For huge files (> 32 MB), sample 64 evenly-spaced 512 KB slices (= 32 MB)
    // so vocabulary discovery stays < 1s and < 100 MB RAM.
    std::unordered_map<std::string, std::uint32_t, SvHash, SvEq> word_counts;
    std::unordered_map<std::string, std::uint32_t, SvHash, SvEq> struct_counts;
    word_counts.reserve(32768u);
    struct_counts.reserve(4096u);

    std::string lower_tmp;
    lower_tmp.reserve(32u);

    auto scan_range = [&](std::size_t start, std::size_t end) {
        std::size_t i = start;
        while (i < end) {
            if (is_ascii_alpha(raw[i])) {
                std::size_t j = i + 1u;
                while (j < end && is_ascii_alpha(raw[j])) ++j;
                const std::size_t len = j - i;
                if (len >= 2u && len <= 24u) {
                    const int cls = classify_word(raw + i, len, lower_tmp);
                    if (cls <= 2) {
                        auto it = word_counts.find(std::string_view(lower_tmp));
                        if (it != word_counts.end()) {
                            ++it->second;
                        } else if (word_counts.size() < 400000u) {
                            word_counts.emplace(lower_tmp, 1u);
                        }
                    }
                }
                i = j;
                continue;
            }
            const std::size_t slen = match_struct_span(raw, end, i);
            if (slen >= 2u && slen <= kMaxTokLen) {
                std::string_view sv(reinterpret_cast<const char*>(raw + i), slen);
                auto it = struct_counts.find(sv);
                if (it != struct_counts.end()) {
                    ++it->second;
                } else if (struct_counts.size() < 65536u) {
                    struct_counts.emplace(std::string(sv), 1u);
                }
                i += slen;
                continue;
            }
            ++i;
        }
    };

    double sample_scale = 1.0;
    constexpr std::size_t kMaxFullScan = 32u << 20;
    if (n <= kMaxFullScan) {
        scan_range(0u, n);
    } else {
        constexpr std::size_t kSlices = 64u;
        constexpr std::size_t kSliceLen = 512u << 10;
        const std::size_t stride = (n - kSliceLen) / (kSlices - 1u);
        for (std::size_t s = 0; s < kSlices; ++s) {
            const std::size_t st = s * stride;
            scan_range(st, std::min(n, st + kSliceLen));
        }
        sample_scale = static_cast<double>(n) / static_cast<double>(kSlices * kSliceLen);
    }

    // Pass 2: Rank candidates by net byte savings.
    struct Cand {
        std::string tok;
        std::uint64_t score;
        bool is_word;
    };
    std::vector<Cand> cands;
    cands.reserve(word_counts.size() + struct_counts.size());

    auto add_cands = [&](const auto& map, bool is_word) {
        for (const auto& [tok, raw_cnt] : map) {
            const std::uint64_t est_cnt =
                static_cast<std::uint64_t>(static_cast<double>(raw_cnt) * sample_scale);
            const std::size_t len = tok.size();
            // Strict net-savings gate so WRT only replaces words that beat a strong CM!
            if (est_cnt < 16u) continue;
            if (len == 2u && est_cnt < 128u) continue;
            if (len == 3u && est_cnt < 48u) continue;
            const std::uint64_t code_est = (est_cnt >= 256u) ? 1u : 2u;
            if (len <= code_est) continue;
            const std::uint64_t gross = est_cnt * (len - code_est);
            const std::uint64_t dict_cost = static_cast<std::uint64_t>(len + 6u) * 3u;
            if (gross <= dict_cost + 32u) continue;
            cands.push_back({tok, gross - dict_cost, is_word});
        }
    };
    add_cands(word_counts, true);
    add_cands(struct_counts, false);

    if (cands.size() < 16u) return res;

    // Cap vocabulary so 1-byte and 2-byte codes dominate (3-byte codes only on >= 16 MB).
    const std::size_t vocab_cap =
        (n <= (2u << 20)) ? 1024u
        : ((n <= (16u << 20)) ? (kTier1Cap + kTier2Cap) : kMaxVocab);
    const std::size_t keep = std::min(cands.size(), vocab_cap);
    std::partial_sort(
        cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(keep), cands.end(),
        [](const Cand& a, const Cand& b) { return a.score > b.score; });
    cands.resize(keep);

    // Split into 3 code-length tiers and sort EACH tier alphabetically so:
    //   1) Front-coding compresses the dictionary by ~3x
    //   2) Morphologically related words share identical lead bytes!
    const std::size_t n1 = std::min(keep, kTier1Cap);
    const std::size_t n2 = std::min(keep - n1, kTier2Cap);
    const std::size_t n3 = std::min(keep - n1 - n2, kTier3Cap);

    std::vector<std::string> tier1, tier2, tier3;
    tier1.reserve(n1);
    tier2.reserve(n2);
    tier3.reserve(n3);

    // Separate flags for whether a token is alphabetic word vs structural
    std::unordered_map<std::string, std::uint32_t, SvHash, SvEq> word_id;
    std::unordered_map<std::string, std::uint32_t, SvHash, SvEq> struct_id;
    word_id.reserve(keep);
    struct_id.reserve(4096u);

    auto collect_tier = [&](std::size_t beg, std::size_t cnt, std::uint32_t id_base,
                            std::vector<std::string>& out_t) {
        std::vector<std::pair<std::string, bool>> tmp;
        tmp.reserve(cnt);
        for (std::size_t k = 0; k < cnt; ++k) {
            tmp.emplace_back(std::move(cands[beg + k].tok), cands[beg + k].is_word);
        }
        std::sort(tmp.begin(), tmp.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (std::size_t k = 0; k < tmp.size(); ++k) {
            const std::uint32_t id = id_base + static_cast<std::uint32_t>(k);
            if (tmp[k].second) {
                word_id.emplace(tmp[k].first, id);
            } else {
                struct_id.emplace(tmp[k].first, id);
            }
            out_t.push_back(std::move(tmp[k].first));
        }
    };
    collect_tier(0u, n1, 0u, tier1);
    collect_tier(n1, n2, static_cast<std::uint32_t>(kTier1Cap), tier2);
    collect_tier(n1 + n2, n3, static_cast<std::uint32_t>(kTier1Cap + kTier2Cap), tier3);

    // Pass 3: Transform `raw[0..n-1]` into `res.stream` with zero-overhead
    // implicit space elision between a dictionary alpha word and any following
    // space + ASCII letter!
    res.stream.reserve((n * 3u) / 4u + 64u);
    std::size_t i = 0u;
    while (i < n) {
        const std::uint8_t b = raw[i];
        if (is_ascii_alpha(b)) {
            std::size_t j = i + 1u;
            while (j < n && is_ascii_alpha(raw[j])) ++j;
            const std::size_t len = j - i;
            if (len >= 2u && len <= 24u) {
                const int cls = classify_word(raw + i, len, lower_tmp);
                if (cls <= 2) {
                    auto it = word_id.find(std::string_view(lower_tmp));
                    if (it != word_id.end()) {
                        if (cls == 1) res.stream.push_back(kCapFirst);
                        else if (cls == 2) res.stream.push_back(kCapAll);
                        emit_code(res.stream, it->second);
                        i = j;
                        // Elide single space before a following ASCII word/letter!
                        if (i + 1u < n && raw[i] == ' ' && is_ascii_alpha(raw[i + 1u])) {
                            ++i;
                        }
                        continue;
                    }
                }
            }
            for (std::size_t k = i; k < j; ++k) {
                res.stream.push_back(raw[k]);
            }
            i = j;
            continue;
        }

        const std::size_t slen = match_struct_span(raw, n, i);
        if (slen >= 2u && slen <= kMaxTokLen) {
            std::string_view sv(reinterpret_cast<const char*>(raw + i), slen);
            auto it = struct_id.find(sv);
            if (it != struct_id.end()) {
                emit_code(res.stream, it->second);
                i += slen;
                continue;
            }
        }

        if (b >= 0x80u) {
            res.stream.push_back(kEscRaw);
        }
        res.stream.push_back(b);
        ++i;
    }

    // Serialize dictionary with front-coding.
    res.dict_blob.reserve(keep * 8u + 32u);
    serialize_tier(res.dict_blob, tier1);
    serialize_tier(res.dict_blob, tier2);
    serialize_tier(res.dict_blob, tier3);

    // Only keep WRT if it substantially shrinks the stream (+ estimated dict cost).
    const std::size_t est_dict_comp = res.dict_blob.size() / 2u;
    if (res.stream.size() + est_dict_comp < (n * 88u) / 100u) {
        res.beneficial = true;
    }
    return res;
}

void wrt_inverse(const std::uint8_t* dict_blob, std::size_t dict_len,
                 const std::uint8_t* stream, std::size_t stream_len,
                 std::uint8_t* dst, std::size_t raw_size) {
    std::size_t off = 0u;
    std::vector<std::string> tier1, tier2, tier3;
    deserialize_tier(dict_blob, dict_len, off, kTier1Cap, tier1);
    deserialize_tier(dict_blob, dict_len, off, kTier2Cap, tier2);
    deserialize_tier(dict_blob, dict_len, off, kTier3Cap, tier3);

    std::size_t out_pos = 0u;
    int cap_mode = 0;  // 0 = none, 1 = first, 2 = all
    bool prev_was_dict_word = false;

    auto emit_word = [&](const std::string& w) {
        const bool is_w = is_word_token(w);
        if (prev_was_dict_word && is_w) {
            if (out_pos >= raw_size) {
                throw std::runtime_error("FZAC WRT implicit space overflow");
            }
            dst[out_pos++] = ' ';
        }
        if (w.size() > raw_size - out_pos) {
            throw std::runtime_error("FZAC WRT decode overflow");
        }
        if (cap_mode == 0) {
            std::memcpy(dst + out_pos, w.data(), w.size());
        } else if (cap_mode == 1) {
            dst[out_pos] = to_upper_ascii(static_cast<std::uint8_t>(w[0]));
            if (w.size() > 1u) {
                std::memcpy(dst + out_pos + 1u, w.data() + 1u, w.size() - 1u);
            }
        } else {
            for (std::size_t k = 0; k < w.size(); ++k) {
                dst[out_pos + k] = to_upper_ascii(static_cast<std::uint8_t>(w[k]));
            }
        }
        out_pos += w.size();
        cap_mode = 0;
        prev_was_dict_word = is_w;
    };

    std::size_t in_pos = 0u;
    while (in_pos < stream_len) {
        const std::uint8_t b = stream[in_pos++];
        if (b < 0x80u) {
            if (prev_was_dict_word && is_ascii_alpha(b)) {
                if (out_pos >= raw_size) {
                    throw std::runtime_error("FZAC WRT implicit space overflow");
                }
                dst[out_pos++] = ' ';
            }
            if (out_pos >= raw_size) {
                throw std::runtime_error("FZAC WRT literal overflow");
            }
            dst[out_pos++] = b;
            cap_mode = 0;
            prev_was_dict_word = false;
        } else if (b == kEscRaw) {
            if (in_pos >= stream_len || out_pos >= raw_size) {
                throw std::runtime_error("FZAC WRT truncated escape");
            }
            dst[out_pos++] = stream[in_pos++];
            cap_mode = 0;
            prev_was_dict_word = false;
        } else if (b == kCapFirst) {
            cap_mode = 1;
        } else if (b == kCapAll) {
            cap_mode = 2;
        } else if (b < kTier2Base) {
            const std::size_t id = static_cast<std::size_t>(b - kTier1Base);
            if (id >= tier1.size()) {
                throw std::runtime_error("FZAC WRT invalid tier1 id");
            }
            emit_word(tier1[id]);
        } else if (b < kTier3Base) {
            if (in_pos >= stream_len) {
                throw std::runtime_error("FZAC WRT truncated tier2 code");
            }
            const std::uint8_t b2 = stream[in_pos++];
            if (b2 < 0x80u) {
                throw std::runtime_error("FZAC WRT invalid tier2 trail byte");
            }
            const std::size_t id =
                static_cast<std::size_t>(b - kTier2Base) * 128u +
                static_cast<std::size_t>(b2 & 0x7Fu);
            if (id >= tier2.size()) {
                throw std::runtime_error("FZAC WRT invalid tier2 id");
            }
            emit_word(tier2[id]);
        } else if (b < 0xFFu) {
            if (in_pos + 1u >= stream_len) {
                throw std::runtime_error("FZAC WRT truncated tier3 code");
            }
            const std::uint8_t b2 = stream[in_pos++];
            const std::uint8_t b3 = stream[in_pos++];
            if (b2 < 0x80u || b3 < 0x80u) {
                throw std::runtime_error("FZAC WRT invalid tier3 trail byte");
            }
            const std::size_t id =
                static_cast<std::size_t>(b - kTier3Base) * 16384u +
                static_cast<std::size_t>(b2 & 0x7Fu) * 128u +
                static_cast<std::size_t>(b3 & 0x7Fu);
            if (id >= tier3.size()) {
                throw std::runtime_error("FZAC WRT invalid tier3 id");
            }
            emit_word(tier3[id]);
        } else {
            throw std::runtime_error("FZAC WRT invalid code 0xFF");
        }
    }

    if (out_pos != raw_size) {
        throw std::runtime_error("FZAC WRT decoded size mismatch");
    }
}

}  // namespace fzip::fzac_v3
