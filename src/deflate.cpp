// fzip — DEFLATE encoder/decoder implementation (RFC 1951).
#include "deflate.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "codec.hpp"

namespace fzip {

namespace {

// --------------------------------------------------------------------------
// Bit writer: 64-bit LSB-first accumulator with 32-bit buffered flushes.
// --------------------------------------------------------------------------
class BitWriter {
  public:
    void reserve(std::size_t bytes) { out_.reserve(bytes); }

    void put_bits(std::uint32_t value, int n) {
        // n <= 24 guaranteed by callers; acc_bits_ < 32 on entry -> <= 56 < 64.
        acc_ |= (static_cast<std::uint64_t>(value) & ((1ULL << n) - 1ULL)) << acc_bits_;
        acc_bits_ += n;
        if (acc_bits_ >= 32) {
            const std::size_t pos = out_.size();
            out_.resize(pos + 4);
            const std::uint32_t lo = static_cast<std::uint32_t>(acc_ & 0xFFFFFFFFULL);
            if constexpr (std::endian::native == std::endian::little) {
                std::memcpy(out_.data() + pos, &lo, 4);
            } else {
                out_[pos]     = static_cast<std::byte>(lo & 0xFFu);
                out_[pos + 1] = static_cast<std::byte>((lo >> 8) & 0xFFu);
                out_[pos + 2] = static_cast<std::byte>((lo >> 16) & 0xFFu);
                out_[pos + 3] = static_cast<std::byte>((lo >> 24) & 0xFFu);
            }
            acc_ >>= 32;
            acc_bits_ -= 32;
        }
    }

    // Put a pre-reversed Huffman code of `len` bits.
    void put_huff(std::uint32_t rev_code, int len) {
        put_bits(rev_code, len);
    }

    void align_to_byte() {
        while (acc_bits_ > 0) {
            out_.push_back(static_cast<std::byte>(acc_ & 0xFFu));
            acc_ >>= 8;
            acc_bits_ -= 8;
        }
        acc_ = 0;
        acc_bits_ = 0;
    }

    void put_bytes(std::span<const std::byte> b) {
        align_to_byte();
        if (!b.empty()) {
            out_.insert(out_.end(), b.begin(), b.end());
        }
    }

    auto data() const -> std::span<const std::byte> {
        return std::span<const std::byte>{out_.data(), out_.size()};
    }
    auto take_data() -> std::vector<std::byte> { return std::move(out_); }
    auto size() const -> std::size_t { return out_.size(); }
    auto total_bits() const -> std::size_t {
        return out_.size() * 8 + static_cast<std::size_t>(acc_bits_);
    }

  private:
    std::vector<std::byte> out_;
    std::uint64_t acc_ = 0;
    int acc_bits_ = 0;
};

// --------------------------------------------------------------------------
// 8-bit reversal LUT and canonical Huffman code construction (RFC 1951 §3.2.2).
// --------------------------------------------------------------------------
constexpr auto make_bitrev8() -> std::array<std::uint8_t, 256> {
    std::array<std::uint8_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
        std::uint8_t r = 0;
        for (int b = 0; b < 8; ++b) {
            r = static_cast<std::uint8_t>((r << 1) | ((i >> b) & 1));
        }
        t[static_cast<std::size_t>(i)] = r;
    }
    return t;
}
constexpr std::array<std::uint8_t, 256> kBitRev8 = make_bitrev8();

constexpr auto reverse_code(std::uint32_t code, int len) -> std::uint32_t {
    const std::uint32_t rev16 =
        (static_cast<std::uint32_t>(kBitRev8[code & 0xFFu]) << 8) |
        static_cast<std::uint32_t>(kBitRev8[(code >> 8) & 0xFFu]);
    return rev16 >> (16 - len);
}

struct HuffCodes {
    std::array<std::uint8_t, 288> lengths{};
    std::array<std::uint16_t, 288> rev_codes{};
    int max_sym = 0;
};

template <std::size_t N>
constexpr auto build_canonical_codes(const std::array<int, N>& lengths, int max_sym)
    -> HuffCodes {
    static_assert(N <= 288);
    HuffCodes h{};
    h.max_sym = max_sym;
    std::array<int, 16> bl_count{};
    for (int s = 0; s < max_sym; ++s) {
        const int len = lengths[static_cast<std::size_t>(s)];
        h.lengths[static_cast<std::size_t>(s)] = static_cast<std::uint8_t>(len);
        if (len > 0 && len <= 15) bl_count[static_cast<std::size_t>(len)]++;
    }
    std::array<std::uint32_t, 16> next_code{};
    std::uint32_t code = 0;
    for (int bits = 1; bits <= 15; ++bits) {
        code = (code + static_cast<std::uint32_t>(bl_count[static_cast<std::size_t>(bits - 1)])) << 1;
        next_code[static_cast<std::size_t>(bits)] = code;
    }
    for (int s = 0; s < max_sym; ++s) {
        const int len = lengths[static_cast<std::size_t>(s)];
        if (len > 0 && len <= 15) {
            const std::uint32_t c = next_code[static_cast<std::size_t>(len)]++;
            h.rev_codes[static_cast<std::size_t>(s)] =
                static_cast<std::uint16_t>(reverse_code(c, len));
        }
    }
    return h;
}

constexpr auto make_fixed_lit_codes() -> HuffCodes {
    std::array<int, 288> lit_lens{};
    for (int s = 0; s <= 143; ++s) lit_lens[static_cast<std::size_t>(s)] = 8;
    for (int s = 144; s <= 255; ++s) lit_lens[static_cast<std::size_t>(s)] = 9;
    for (int s = 256; s <= 279; ++s) lit_lens[static_cast<std::size_t>(s)] = 7;
    for (int s = 280; s <= 287; ++s) lit_lens[static_cast<std::size_t>(s)] = 8;
    return build_canonical_codes(lit_lens, 288);
}

constexpr auto make_fixed_dist_codes() -> HuffCodes {
    std::array<int, 32> dist_lens{};
    for (int s = 0; s < 30; ++s) dist_lens[static_cast<std::size_t>(s)] = 5;
    return build_canonical_codes(dist_lens, 30);
}

constexpr HuffCodes kFixedLitCodes = make_fixed_lit_codes();
constexpr HuffCodes kFixedDistCodes = make_fixed_dist_codes();

// Fast canonical Huffman decoder with a 9-bit primary LUT.
struct HuffDecoder {
    std::array<std::uint16_t, 512> fast_lut{};  // sym | (len << 12), 0 if len > 9
    std::array<int, 16> count{};
    std::array<int, 16> first_code{};
    std::array<int, 16> first_index{};
    std::array<int, 288> symbols{};
    int max_len = 0;
};

template <std::size_t N>
auto build_huff_decoder(const std::array<int, N>& lengths, int max_sym)
    -> HuffDecoder {
    HuffDecoder d;
    for (int s = 0; s < max_sym; ++s) {
        int len = lengths[static_cast<std::size_t>(s)];
        if (len > 0 && len <= 15) d.count[static_cast<std::size_t>(len)]++;
    }
    int code = 0;
    int idx = 0;
    std::array<int, 16> next_code{};
    for (int len = 1; len <= 15; ++len) {
        d.first_code[static_cast<std::size_t>(len)] = code;
        next_code[static_cast<std::size_t>(len)] = code;
        d.first_index[static_cast<std::size_t>(len)] = idx;
        code = (code + d.count[static_cast<std::size_t>(len)]) << 1;
        idx += d.count[static_cast<std::size_t>(len)];
        if (d.count[static_cast<std::size_t>(len)] > 0) d.max_len = len;
    }
    std::array<int, 16> next = d.first_index;
    for (int s = 0; s < max_sym; ++s) {
        int len = lengths[static_cast<std::size_t>(s)];
        if (len > 0 && len <= 15) {
            d.symbols[static_cast<std::size_t>(next[static_cast<std::size_t>(len)]++)] = s;
            int c = next_code[static_cast<std::size_t>(len)]++;
            if (len <= 9) {
                std::uint32_t rev = reverse_code(static_cast<std::uint32_t>(c), len);
                std::uint16_t val = static_cast<std::uint16_t>(s | (len << 12));
                int step = 1 << len;
                for (int fill = static_cast<int>(rev); fill < 512; fill += step) {
                    d.fast_lut[static_cast<std::size_t>(fill)] = val;
                }
            }
        }
    }
    return d;
}

// Length-limited prefix code lengths via package-merge (Larmore-Hirschberg),
// zero heap allocations using fixed-size stack buffers (num_codes <= 288).
auto package_merge_sorted(int max_length, int num_codes, std::uint64_t* A) -> int {
    if (num_codes <= 0 || max_length == 0) return 0;
    if (num_codes <= 2) {
        A[0] = 1;
        if (num_codes == 2) A[1] = 1;
        return 1;
    }
    if (max_length > 63) return 0;
    if ((1ULL << max_length) < static_cast<std::uint64_t>(num_codes)) return 0;

    constexpr int kMaxBuf = 576;
    std::uint64_t buf_a[kMaxBuf];
    std::uint64_t buf_b[kMaxBuf];
    std::uint64_t is_merged[kMaxBuf]{};
    std::uint64_t* current = buf_a;
    std::uint64_t* previous = buf_b;
    for (int i = 0; i < num_codes; ++i) previous[i] = A[i];
    int num_previous = num_codes;
    const int num_relevant = 2 * num_codes - 2;

    std::uint64_t mask = 1;
    for (int bits = max_length - 1; bits > 0; --bits) {
        num_previous &= ~1;
        current[0] = A[0];
        current[1] = A[1];
        std::uint64_t sum = current[0] + current[1];
        int num_current = 2;
        int num_hist = 2;
        int num_merged = 0;
        for (;;) {
            if (num_hist < num_codes && A[num_hist] <= sum) {
                current[num_current++] = A[num_hist++];
                continue;
            }
            is_merged[num_current] |= mask;
            current[num_current] = sum;
            ++num_current;
            ++num_merged;
            if (num_merged * 2 >= num_previous) break;
            sum = previous[num_merged * 2] + previous[num_merged * 2 + 1];
        }
        while (num_hist < num_codes) current[num_current++] = A[num_hist++];
        mask <<= 1;

        if (num_previous >= num_relevant) {
            bool keep_going = false;
            for (int i = num_relevant - 1; i > 0; --i) {
                if (previous[i] != current[i]) { keep_going = true; break; }
            }
            if (!keep_going) break;
        }
        std::swap(previous, current);
        num_previous = num_current;
    }
    mask >>= 1;

    unsigned int code_lengths[288]{};
    int num_analyze = num_relevant;
    while (mask != 0) {
        int num_merged = 0;
        code_lengths[0]++;
        code_lengths[1]++;
        int symbol = 2;
        for (int i = symbol; i < num_analyze; ++i) {
            if ((is_merged[i] & mask) == 0) {
                code_lengths[symbol]++;
                ++symbol;
            } else {
                ++num_merged;
            }
        }
        num_analyze = 2 * num_merged;
        mask >>= 1;
    }
    for (int i = 0; i < num_analyze; ++i) code_lengths[i]++;
    for (int i = 0; i < num_codes; ++i) A[i] = code_lengths[i];
    return static_cast<int>(code_lengths[0]);
}

template <std::size_t N>
auto build_huff_lengths(const std::array<std::uint32_t, N>& freqs,
                        int max_sym, int max_len) -> std::array<int, N> {
    static_assert(N <= 288);
    std::array<int, N> lengths{};
    struct Sym {
        std::uint64_t f;
        int s;
    };
    Sym syms[N];
    int num_syms = 0;
    for (int s = 0; s < max_sym; ++s) {
        if (freqs[static_cast<std::size_t>(s)] > 0) {
            syms[num_syms++] = {freqs[static_cast<std::size_t>(s)], s};
        }
    }
    if (num_syms == 0) return lengths;
    std::sort(syms, syms + num_syms, [](const Sym& a, const Sym& b) {
        if (a.f != b.f) return a.f < b.f;
        return a.s < b.s;
    });
    std::uint64_t hist[N];
    for (int i = 0; i < num_syms; ++i) hist[i] = syms[i].f;
    package_merge_sorted(max_len, num_syms, hist);
    for (int i = 0; i < num_syms; ++i) {
        lengths[static_cast<std::size_t>(syms[i].s)] = static_cast<int>(hist[i]);
    }
    return lengths;
}

// --------------------------------------------------------------------------
// Length & distance LUTs (RFC 1951 §3.2.5).
// --------------------------------------------------------------------------
struct LenCode { int sym; int extra; int base; };
struct DistCode { int sym; int extra; int base; };

constexpr auto init_len_table() -> std::array<LenCode, 259> {
    std::array<LenCode, 259> table{};
    struct Row { int min; int max; int sym; int extra; int base; };
    constexpr Row tab[] = {
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
    for (const auto& e : tab) {
        for (int l = e.min; l <= e.max; ++l) {
            table[static_cast<std::size_t>(l)] = {e.sym, e.extra, e.base};
        }
    }
    return table;
}

auto init_dist_table() -> std::array<DistCode, 32769> {
    std::array<DistCode, 32769> table{};
    struct Row { int min; int max; int sym; int extra; int base; };
    static constexpr Row tab[] = {
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
    for (const auto& e : tab) {
        for (int d = e.min; d <= e.max; ++d) {
            table[static_cast<std::size_t>(d)] = {e.sym, e.extra, e.base};
        }
    }
    return table;
}

constexpr auto kLenTable = init_len_table();
const auto kDistTable = init_dist_table();

constexpr int kLenExtraBySym[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
    2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr int kDistExtraBySym[30] = {
    0, 0, 0, 0, 1, 1, 2,  2,  3,  3,  4,  4,  5,  5,  6,
    6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
// Upper endpoints of the 29 length symbol buckets (used to prune long-match DP).
constexpr int kLenBucketMax[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 22, 26, 30,
    34, 42, 50, 58, 66, 82, 98, 114, 130, 162, 194, 226, 257, 258};

inline auto length_code(int len) -> LenCode {
    return kLenTable[static_cast<std::size_t>(len)];
}
inline auto length_to_symbol(int len) -> int {
    return kLenTable[static_cast<std::size_t>(len)].sym - 257;
}
inline auto distance_code(int dist) -> DistCode {
    return kDistTable[static_cast<std::size_t>(dist)];
}
inline auto distance_to_symbol(int dist) -> int {
    return kDistTable[static_cast<std::size_t>(dist)].sym;
}

// --------------------------------------------------------------------------
// Compact 4-byte Token and 4-byte MatchCand.
// --------------------------------------------------------------------------
struct Token {
    std::uint32_t raw = 0;

    static constexpr auto make_lit(std::uint8_t c) -> Token {
        return Token{static_cast<std::uint32_t>(c)};
    }
    static constexpr auto make_match(int len, int dist) -> Token {
        return Token{0x80000000u |
                     (static_cast<std::uint32_t>(len) << 16) |
                     static_cast<std::uint32_t>(dist)};
    }
    constexpr auto is_match() const -> bool { return (raw & 0x80000000u) != 0; }
    constexpr auto lit() const -> std::uint8_t { return static_cast<std::uint8_t>(raw & 0xFFu); }
    constexpr auto len() const -> int { return static_cast<int>((raw >> 16) & 0x1FFu); }
    constexpr auto dist() const -> int { return static_cast<int>(raw & 0xFFFFu); }
};
static_assert(sizeof(Token) == 4);

struct MatchCand {
    std::uint16_t len;
    std::uint16_t dist;
};
static_assert(sizeof(MatchCand) == 4);

struct Match {
    int dist = 0;
    int len = 0;
};

// --------------------------------------------------------------------------
// Thread-local epoch-reset workspace (zero heap alloc / memset per block).
// --------------------------------------------------------------------------
constexpr int kWindow = 32768;
constexpr int kMinMatch = 3;
constexpr int kMaxMatch = 258;
constexpr int kHashBits = 16;
constexpr int kHashSize = 1 << kHashBits;
constexpr std::uint32_t kHashMask = kHashSize - 1;

struct DeflateWorkspace {
    std::array<std::int32_t, kHashSize> head3{};
    std::array<std::int32_t, kWindow> prev3{};
    std::array<std::int32_t, kHashSize> head4{};
    std::array<std::int32_t, kWindow> prev4{};
    std::array<std::int32_t, kHashSize> bt_left{};
    std::array<std::int32_t, kHashSize> bt_right{};
    std::int32_t epoch_base = 1;

    auto begin_block(std::size_t full_size) -> std::int32_t {
        if (epoch_base + static_cast<std::int64_t>(full_size) + 64 >= 0x3FFFFFFFLL) {
            head3.fill(0);
            head4.fill(0);
            epoch_base = 1;
        }
        const std::int32_t base = epoch_base;
        epoch_base += static_cast<std::int32_t>(full_size) + 1;
        return base;
    }
};

auto get_workspace() -> DeflateWorkspace& {
    thread_local std::unique_ptr<DeflateWorkspace> ws;
    if (!ws) ws = std::make_unique<DeflateWorkspace>();
    return *ws;
}

inline auto load_u32(const std::uint8_t* p) -> std::uint32_t {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline auto hash4(const std::uint8_t* p) -> std::uint32_t {
    return (load_u32(p) * 2654435761u) >> (32 - kHashBits);
}

inline auto hash3(const std::uint8_t* p) -> std::uint32_t {
    std::uint32_t h = (static_cast<std::uint32_t>(p[0]) << 16) |
                      (static_cast<std::uint32_t>(p[1]) << 8) |
                      static_cast<std::uint32_t>(p[2]);
    return ((h * 2654435761u) >> (32 - kHashBits)) & kHashMask;
}

inline auto match_len_fast(const std::uint8_t* a, const std::uint8_t* b,
                           int maxl) -> int {
    int l = 0;
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

inline void insert_hash3(const std::uint8_t* data, std::size_t size,
                         std::size_t pos, DeflateWorkspace& ws,
                         std::int32_t base) {
    if (pos + kMinMatch > size) return;
    const std::uint32_t h = hash3(data + pos);
    const std::int32_t tag = base + static_cast<std::int32_t>(pos);
    ws.prev3[pos & (kWindow - 1)] = ws.head3[h];
    ws.head3[h] = tag;
}

inline void insert_hash4(const std::uint8_t* data, std::size_t size,
                         std::size_t pos, DeflateWorkspace& ws,
                         std::int32_t base) {
    if (pos + 4 > size) return;
    const std::uint32_t h = hash4(data + pos);
    const std::int32_t tag = base + static_cast<std::int32_t>(pos);
    ws.prev4[pos & (kWindow - 1)] = ws.head4[h];
    ws.head4[h] = tag;
}

// Find best match at `pos` using hash chains (levels 1-8).
auto find_match_hc(const std::uint8_t* data, std::size_t size, std::size_t pos,
                   DeflateWorkspace& ws, std::int32_t base, int max_effort,
                   int nice_len, bool use_h3) -> Match {
    Match best;
    if (pos + kMinMatch > size) return best;
    const int maxl = static_cast<int>(std::min<std::size_t>(kMaxMatch, size - pos));
    const int eff_nice = std::min(nice_len, maxl);
    const std::size_t min_p = (pos > kWindow) ? (pos - kWindow) : 0;
    const std::int32_t limit_tag = base + static_cast<std::int32_t>(min_p);

    if (pos + 4 <= size) {
        const std::uint32_t p4 = load_u32(data + pos);
        std::int32_t cand_tag = ws.head4[hash4(data + pos)];
        int tries = max_effort;
        while (cand_tag >= limit_tag && tries-- > 0) {
            const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
            if (best.len >= 4 && best.len < maxl &&
                (data[cand + static_cast<std::size_t>(best.len)] !=
                     data[pos + static_cast<std::size_t>(best.len)] ||
                 data[cand + static_cast<std::size_t>(best.len) - 1] !=
                     data[pos + static_cast<std::size_t>(best.len) - 1])) {
                cand_tag = ws.prev4[cand & (kWindow - 1)];
                continue;
            }
            if (load_u32(data + cand) != p4) {
                cand_tag = ws.prev4[cand & (kWindow - 1)];
                continue;
            }
            const int l = match_len_from(data + cand, data + pos, 4, maxl);
            const int d = static_cast<int>(pos - cand);
            if (l > best.len) {
                // Avoid jumping to a vastly more distant match for just +1 byte.
                if (best.len >= 4 && l == best.len + 1 &&
                    kDistExtraBySym[static_cast<std::size_t>(distance_to_symbol(d))] >
                        kDistExtraBySym[static_cast<std::size_t>(distance_to_symbol(best.dist))] + 7) {
                    cand_tag = ws.prev4[cand & (kWindow - 1)];
                    continue;
                }
                best.len = l;
                best.dist = d;
                if (l >= eff_nice) return best;
            }
            cand_tag = ws.prev4[cand & (kWindow - 1)];
        }
    }

    if (use_h3 && (best.len < 4 || (best.len == 4 && best.dist > 4096))) {
        std::int32_t cand_tag = ws.head3[hash3(data + pos)];
        int tries = std::min(max_effort, 16);
        const std::uint8_t b0 = data[pos];
        const std::uint8_t b1 = data[pos + 1];
        const std::uint8_t b2 = data[pos + 2];
        while (cand_tag >= limit_tag && tries-- > 0) {
            const std::size_t cand = static_cast<std::size_t>(cand_tag - base);
            const int d = static_cast<int>(pos - cand);
            if (d > 4096) break;
            if (data[cand] == b0 && data[cand + 1] == b1 && data[cand + 2] == b2) {
                const int l = match_len_from(data + cand, data + pos, 3, maxl);
                if (best.len < 4 || (l == 3 && d < 512 && best.dist > 4096) || l > best.len) {
                    best.len = l;
                    best.dist = d;
                }
                break;
            }
            cand_tag = ws.prev3[cand & (kWindow - 1)];
        }
    }
    return best;
}

// Score difference for lazy matching in approx bits.
inline auto lazy_beats(const Match& a, const Match& b, int margin) -> bool {
    if (a.len < kMinMatch) return false;
    const int ea = kDistExtraBySym[static_cast<std::size_t>(distance_to_symbol(a.dist))];
    const int eb = kDistExtraBySym[static_cast<std::size_t>(distance_to_symbol(b.dist))];
    const int score_a = a.len * 7 - ea;
    const int score_b = b.len * 7 - eb;
    return score_a > score_b + margin;
}

auto lz77_encode(const std::uint8_t* data, std::size_t size, int level,
                 std::array<std::uint32_t, 288>& lit_freq,
                 std::array<std::uint32_t, 32>& dist_freq,
                 std::size_t hist = 0) -> std::vector<Token> {
    std::vector<Token> tokens;
    tokens.reserve(size / 2 + 16);
    lit_freq.fill(0);
    dist_freq.fill(0);

    const auto* full_data = data - hist;
    const std::size_t full_size = size + hist;

    DeflateWorkspace& ws = get_workspace();
    const std::int32_t base = ws.begin_block(full_size);

    int effort = 16;
    int nice_len = 64;
    int lazy_depth = 1;
    int max_lazy = 32;
    int good_len = 12;
    int skip_len = 0;
    bool use_h3 = true;
    switch (level) {
        case 1: effort = 4;    nice_len = 16;  lazy_depth = 0; max_lazy = 0;   good_len = 8;  skip_len = 8;  use_h3 = false; break;
        case 2: effort = 8;    nice_len = 32;  lazy_depth = 0; max_lazy = 0;   good_len = 16; skip_len = 16; use_h3 = true;  break;
        case 3: effort = 24;   nice_len = 48;  lazy_depth = 0; max_lazy = 0;   good_len = 24; skip_len = 32; use_h3 = true;  break;
        case 4: effort = 16;   nice_len = 48;  lazy_depth = 1; max_lazy = 16;  good_len = 8;   skip_len = 0;  use_h3 = true;  break;
        case 5: effort = 32;   nice_len = 64;  lazy_depth = 1; max_lazy = 32;  good_len = 16;  skip_len = 0;  use_h3 = true;  break;
        case 6: effort = 128;  nice_len = 258; lazy_depth = 2; max_lazy = 128; good_len = 128; skip_len = 0;  use_h3 = true;  break;
        case 7: effort = 256;  nice_len = 258; lazy_depth = 2; max_lazy = 258; good_len = 64;  skip_len = 0;  use_h3 = true;  break;
        default: effort = 1024; nice_len = 258; lazy_depth = 2; max_lazy = 258; good_len = 128; skip_len = 0;  use_h3 = true;  break;
    }

    auto insert_pos = [&](std::size_t p) {
        if (use_h3) insert_hash3(full_data, full_size, p, ws, base);
        insert_hash4(full_data, full_size, p, ws, base);
    };

    std::size_t inserted_up_to = 0;
    auto ensure_inserted = [&](std::size_t target) {
        while (inserted_up_to < target) {
            insert_pos(inserted_up_to++);
        }
    };

    ensure_inserted(hist);

    auto emit_lit = [&](std::uint8_t c) {
        tokens.push_back(Token::make_lit(c));
        lit_freq[c]++;
    };
    auto emit_match = [&](int len, int dist) {
        tokens.push_back(Token::make_match(len, dist));
        lit_freq[static_cast<std::size_t>(257 + length_to_symbol(len))]++;
        dist_freq[static_cast<std::size_t>(distance_to_symbol(dist))]++;
    };

    std::size_t pos = hist;
    while (pos < full_size) {
        ensure_inserted(pos);
        Match m = find_match_hc(full_data, full_size, pos, ws, base, effort,
                                nice_len, use_h3);
        if (m.len < kMinMatch) {
            emit_lit(full_data[pos]);
            ++pos;
            continue;
        }

        if (lazy_depth >= 1) {
            while (pos + 1 < full_size && m.len < max_lazy) {
                const int eff = (m.len >= good_len) ? std::max(4, effort >> 2) : effort;
                ensure_inserted(pos + 1);
                Match m2 = find_match_hc(full_data, full_size, pos + 1, ws, base,
                                         eff, nice_len, use_h3);
                if (lazy_beats(m2, m, 0)) {
                    emit_lit(full_data[pos]);
                    ++pos;
                    m = m2;
                    continue;
                }
                if (lazy_depth >= 2 && pos + 2 < full_size && m.len < (max_lazy >> 1)) {
                    ensure_inserted(pos + 2);
                    Match m3 = find_match_hc(full_data, full_size, pos + 2, ws, base,
                                             eff, nice_len, use_h3);
                    if (lazy_beats(m3, m, 7)) {
                        emit_lit(full_data[pos]);
                        emit_lit(full_data[pos + 1]);
                        pos += 2;
                        m = m3;
                        continue;
                    }
                }
                break;
            }
        }

        emit_match(m.len, m.dist);
        const std::size_t next_pos = pos + static_cast<std::size_t>(m.len);
        if (skip_len > 0 && m.len >= skip_len) {
            if (inserted_up_to <= pos) insert_pos(pos);
            if (next_pos > pos + 2) insert_pos(next_pos - 1);
            inserted_up_to = next_pos;
        } else {
            ensure_inserted(next_pos);
        }
        pos = next_pos;
    }

    lit_freq[256]++;
    return tokens;
}

// --------------------------------------------------------------------------
// BT4 Binary Search Tree match finder + Optimal (shortest-path) LZ77 parse.
// --------------------------------------------------------------------------
constexpr int kMaxCands = 32;

// Insert `pos` into BT4 without collecting candidates (used for `hist` bytes).
void bt_skip_insert(const std::uint8_t* data, std::size_t size, std::size_t pos,
                    DeflateWorkspace& ws, std::int32_t base, int depth,
                    int cmp_cap) {
    insert_hash3(data, size, pos, ws, base);
    if (pos + 4 > size) return;
    const int maxl = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(cmp_cap), size - pos));
    const std::size_t min_p = (pos > kWindow) ? (pos - kWindow) : 0;
    const std::int32_t limit_tag = base + static_cast<std::int32_t>(min_p);

    const std::uint32_t h4 = hash4(data + pos);
    std::int32_t cur_tag = ws.head4[h4];
    ws.head4[h4] = base + static_cast<std::int32_t>(pos);

    const std::size_t pos_slot = pos & kHashMask;
    std::int32_t* ptr_left = &ws.bt_left[pos_slot];
    std::int32_t* ptr_right = &ws.bt_right[pos_slot];
    int len_left = 0;
    int len_right = 0;

    int tries = depth;
    while (cur_tag >= limit_tag && tries-- > 0) {
        const std::size_t cand = static_cast<std::size_t>(cur_tag - base);
        const std::size_t cand_slot = cand & kHashMask;
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

// Find Pareto-optimal matches at `pos` via BT4 + head3 and insert `pos` into both.
auto bt_find_and_insert(const std::uint8_t* data, std::size_t size,
                        std::size_t pos, DeflateWorkspace& ws,
                        std::int32_t base, int depth, int nice_len,
                        MatchCand (&out)[kMaxCands]) -> int {
    int n = 0;
    if (pos + kMinMatch > size) {
        return 0;
    }

    const int maxl = static_cast<int>(std::min<std::size_t>(kMaxMatch, size - pos));
    const std::size_t min_p = (pos > kWindow) ? (pos - kWindow) : 0;
    const std::int32_t limit_tag = base + static_cast<std::int32_t>(min_p);

    // 1. BT4 insert & search for all 4+ byte Pareto matches (visits strictly increasing dist).
    if (pos + 4 <= size) {
        const int eff_nice = std::min(nice_len, maxl);
        const std::uint32_t h4 = hash4(data + pos);
        std::int32_t cur_tag = ws.head4[h4];
        ws.head4[h4] = base + static_cast<std::int32_t>(pos);

        const std::size_t pos_slot = pos & kHashMask;
        std::int32_t* ptr_left = &ws.bt_left[pos_slot];
        std::int32_t* ptr_right = &ws.bt_right[pos_slot];
        int len_left = 0;
        int len_right = 0;
        int best_len = 3;
        int tries = depth;
        bool attached = false;

        while (cur_tag >= limit_tag && tries-- > 0) {
            const std::size_t cand = static_cast<std::size_t>(cur_tag - base);
            const std::size_t cand_slot = cand & kHashMask;
            int len = std::min(len_left, len_right);
            len = match_len_from(data + cand, data + pos, len, eff_nice);

            if (len > best_len) {
                int true_len = len;
                if (len == eff_nice && eff_nice < maxl) {
                    true_len = match_len_from(data + cand, data + pos, eff_nice, maxl);
                }
                const int d = static_cast<int>(pos - cand);
                if (n < kMaxCands) {
                    out[n++] = MatchCand{static_cast<std::uint16_t>(true_len),
                                         static_cast<std::uint16_t>(d)};
                } else {
                    out[n - 1] = MatchCand{static_cast<std::uint16_t>(true_len),
                                           static_cast<std::uint16_t>(d)};
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

    // 2. Check head3 for a closer 3+ byte match (and insert `pos` into head3).
    {
        const std::uint32_t h3 = hash3(data + pos);
        std::int32_t cand_tag = ws.head3[h3];
        const std::uint16_t max_d3 = (n > 0) ? out[0].dist : static_cast<std::uint16_t>(kWindow + 1);
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
                    // Remove any BT4 entries dominated by (l, d).
                    int w = 0;
                    for (int j = 0; j < n; ++j) {
                        if (!(out[j].dist >= d && out[j].len <= l)) {
                            out[w++] = out[j];
                        }
                    }
                    n = w;
                    if (n < kMaxCands) {
                        for (int j = n; j > 0; --j) out[j] = out[j - 1];
                        out[0] = MatchCand{static_cast<std::uint16_t>(l),
                                           static_cast<std::uint16_t>(d)};
                        ++n;
                    } else {
                        for (int j = n - 1; j > 0; --j) out[j] = out[j - 1];
                        out[0] = MatchCand{static_cast<std::uint16_t>(l),
                                           static_cast<std::uint16_t>(d)};
                    }
                    break;
                }
                cand_tag = ws.prev3[cand & (kWindow - 1)];
            }
        }
        ws.prev3[pos & (kWindow - 1)] = ws.head3[h3];
        ws.head3[h3] = base + static_cast<std::int32_t>(pos);
    }

    return n;
}

// --------------------------------------------------------------------------
// Code-length RLE & O(alphabet) block-size estimators + Huffman smoothing.
// --------------------------------------------------------------------------
struct ClRle { int code; int xbits; };

auto build_cl_rle(const std::array<int, 286>& lit_lens, int hlit,
                  const std::array<int, 30>& dist_lens, int hdist,
                  std::array<std::uint32_t, 19>& cl_freq,
                  std::array<ClRle, 320>& rle) -> int {
    int all_lens[320];
    int total = 0;
    for (int i = 0; i < hlit; ++i) all_lens[total++] = lit_lens[static_cast<std::size_t>(i)];
    for (int i = 0; i < hdist; ++i) all_lens[total++] = dist_lens[static_cast<std::size_t>(i)];

    int rle_len = 0;
    cl_freq.fill(0);
    for (int i = 0; i < total;) {
        const int v = all_lens[i];
        int run = 1;
        while (i + run < total && all_lens[i + run] == v) ++run;

        if (v == 0) {
            int emitted = 0;
            while (emitted < run) {
                int rem = run - emitted;
                if (rem >= 11) {
                    int r = std::min(rem, 138);
                    if (rem > 138 && rem - 138 < 3) r = rem - 3;
                    rle[static_cast<std::size_t>(rle_len++)] = {18, r - 11};
                    cl_freq[18]++;
                    emitted += r;
                } else if (rem >= 3) {
                    int r = std::min(rem, 10);
                    rle[static_cast<std::size_t>(rle_len++)] = {17, r - 3};
                    cl_freq[17]++;
                    emitted += r;
                } else {
                    rle[static_cast<std::size_t>(rle_len++)] = {0, 0};
                    cl_freq[0]++;
                    emitted++;
                }
            }
        } else {
            rle[static_cast<std::size_t>(rle_len++)] = {v, 0};
            cl_freq[static_cast<std::size_t>(v)]++;
            int emitted = 1;
            while (emitted < run) {
                int rem = run - emitted;
                if (rem >= 3) {
                    int r = std::min(rem, 6);
                    if (rem > 6 && rem - 6 < 3) r = rem - 3;
                    rle[static_cast<std::size_t>(rle_len++)] = {16, r - 3};
                    cl_freq[16]++;
                    emitted += r;
                } else {
                    rle[static_cast<std::size_t>(rle_len++)] = {v, 0};
                    cl_freq[static_cast<std::size_t>(v)]++;
                    emitted++;
                }
            }
        }
        i += run;
    }
    return rle_len;
}

auto eval_dynamic_bits(const std::array<std::uint32_t, 288>& lit_freq,
                       const std::array<std::uint32_t, 32>& dist_freq,
                       const std::array<int, 286>& lit_lens,
                       const std::array<int, 30>& dist_lens) -> std::size_t {
    int hlit = 286;
    while (hlit > 257 && lit_lens[static_cast<std::size_t>(hlit - 1)] == 0) --hlit;
    int hdist = 30;
    while (hdist > 1 && dist_lens[static_cast<std::size_t>(hdist - 1)] == 0) --hdist;

    std::array<std::uint32_t, 19> cl_freq{};
    std::array<ClRle, 320> rle{};
    build_cl_rle(lit_lens, hlit, dist_lens, hdist, cl_freq, rle);
    auto cl_lens = build_huff_lengths(cl_freq, 19, 7);

    static constexpr int order[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                      11, 4,  12, 3, 13, 2, 14, 1, 15};
    int hclen = 19;
    while (hclen > 4 && cl_lens[static_cast<std::size_t>(order[hclen - 1])] == 0) --hclen;

    std::size_t bits = 3 + 5 + 5 + 4 + static_cast<std::size_t>(hclen) * 3;
    for (int s = 0; s < 19; ++s) {
        const auto f = cl_freq[static_cast<std::size_t>(s)];
        if (f > 0) {
            int extra = (s == 16) ? 2 : (s == 17) ? 3 : (s == 18) ? 7 : 0;
            bits += static_cast<std::size_t>(f) *
                    static_cast<std::size_t>(cl_lens[static_cast<std::size_t>(s)] + extra);
        }
    }
    for (int s = 0; s <= 256; ++s) {
        const auto f = lit_freq[static_cast<std::size_t>(s)];
        if (f > 0) {
            bits += static_cast<std::size_t>(f) *
                    static_cast<std::size_t>(lit_lens[static_cast<std::size_t>(s)]);
        }
    }
    for (int s = 257; s < 286; ++s) {
        const auto f = lit_freq[static_cast<std::size_t>(s)];
        if (f > 0) {
            bits += static_cast<std::size_t>(f) *
                    static_cast<std::size_t>(lit_lens[static_cast<std::size_t>(s)] +
                                             kLenExtraBySym[s - 257]);
        }
    }
    for (int d = 0; d < 30; ++d) {
        const auto f = dist_freq[static_cast<std::size_t>(d)];
        if (f > 0) {
            bits += static_cast<std::size_t>(f) *
                    static_cast<std::size_t>(dist_lens[static_cast<std::size_t>(d)] +
                                             kDistExtraBySym[d]);
        }
    }
    return bits;
}

template <std::size_t N>
void smooth_freqs_for_rle(int length, const std::uint32_t* in_freqs,
                          std::array<std::uint32_t, N>& out) {
    out.fill(0);
    int actual_len = length;
    while (actual_len > 0 && in_freqs[actual_len - 1] == 0) --actual_len;
    if (actual_len == 0) return;
    for (int i = 0; i < actual_len; ++i) {
        out[static_cast<std::size_t>(i)] = in_freqs[i];
    }

    bool good_for_rle[288]{};
    int stride = 0;
    std::uint32_t symbol = out[0];
    for (int i = 0; i < actual_len; ++i) {
        if (out[static_cast<std::size_t>(i)] != symbol) {
            if ((symbol == 0 && stride >= 4) || (symbol != 0 && stride >= 6)) {
                for (int k = i - stride; k < i; ++k) good_for_rle[k] = true;
            }
            stride = 1;
            symbol = out[static_cast<std::size_t>(i)];
        } else {
            ++stride;
        }
    }
    if ((symbol == 0 && stride >= 4) || (symbol != 0 && stride >= 6)) {
        for (int k = actual_len - stride; k < actual_len; ++k) good_for_rle[k] = true;
    }

    stride = 0;
    std::uint64_t sum = 0;
    std::uint32_t limit = out[0];
    for (int i = 0; i <= actual_len; ++i) {
        const bool flush =
            (i == actual_len) || good_for_rle[i] ||
            out[static_cast<std::size_t>(i)] == 0 ||
            (limit > 0 && (out[static_cast<std::size_t>(i)] > limit * 2 ||
                           limit > out[static_cast<std::size_t>(i)] * 2));
        if (flush) {
            if (stride >= 2) {
                std::uint32_t avg = static_cast<std::uint32_t>(
                    (sum + static_cast<std::uint64_t>(stride) / 2) /
                    static_cast<std::uint64_t>(stride));
                if (avg < 1) avg = 1;
                for (int k = i - stride; k < i; ++k) {
                    if (out[static_cast<std::size_t>(k)] > 0) {
                        out[static_cast<std::size_t>(k)] = avg;
                    }
                }
            }
            stride = 0;
            sum = 0;
            if (i < actual_len && !good_for_rle[i] &&
                out[static_cast<std::size_t>(i)] > 0) {
                limit = out[static_cast<std::size_t>(i)];
                stride = 1;
                sum = limit;
            } else {
                limit = 0;
            }
        } else {
            ++stride;
            sum += out[static_cast<std::size_t>(i)];
        }
    }
}

void build_dynamic_lengths(const std::array<std::uint32_t, 288>& lit_freq,
                           const std::array<std::uint32_t, 32>& dist_freq,
                           bool try_rle_opt,
                           std::array<int, 286>& lit_lens,
                           std::array<int, 30>& dist_lens) {
    auto lit_full = build_huff_lengths(lit_freq, 286, 15);
    std::array<std::uint32_t, 32> df{};
    for (int i = 0; i < 30; ++i) df[static_cast<std::size_t>(i)] = dist_freq[static_cast<std::size_t>(i)];
    auto dist_full = build_huff_lengths(df, 30, 15);
    for (int i = 0; i < 286; ++i) lit_lens[static_cast<std::size_t>(i)] = lit_full[static_cast<std::size_t>(i)];
    for (int i = 0; i < 30; ++i) dist_lens[static_cast<std::size_t>(i)] = dist_full[static_cast<std::size_t>(i)];

    if (!try_rle_opt) return;

    const std::size_t base_bits = eval_dynamic_bits(lit_freq, dist_freq, lit_lens, dist_lens);
    std::array<std::uint32_t, 288> smooth_lit{};
    std::array<std::uint32_t, 32> smooth_dist{};
    smooth_freqs_for_rle(286, lit_freq.data(), smooth_lit);
    smooth_freqs_for_rle(30, df.data(), smooth_dist);

    auto opt_lit_full = build_huff_lengths(smooth_lit, 286, 15);
    auto opt_dist_full = build_huff_lengths(smooth_dist, 30, 15);
    std::array<int, 286> opt_lit{};
    std::array<int, 30> opt_dist{};
    for (int i = 0; i < 286; ++i) opt_lit[static_cast<std::size_t>(i)] = opt_lit_full[static_cast<std::size_t>(i)];
    for (int i = 0; i < 30; ++i) opt_dist[static_cast<std::size_t>(i)] = opt_dist_full[static_cast<std::size_t>(i)];

    const std::size_t opt_bits = eval_dynamic_bits(lit_freq, dist_freq, opt_lit, opt_dist);
    if (opt_bits < base_bits) {
        lit_lens = opt_lit;
        dist_lens = opt_dist;
    }
}

auto dynamic_block_bits(const std::array<std::uint32_t, 288>& lit_freq,
                        const std::array<std::uint32_t, 32>& dist_freq)
    -> std::size_t {
    std::array<int, 286> lit_lens{};
    std::array<int, 30> dist_lens{};
    build_dynamic_lengths(lit_freq, dist_freq, false, lit_lens, dist_lens);
    return eval_dynamic_bits(lit_freq, dist_freq, lit_lens, dist_lens);
}

auto fixed_block_bits(const std::array<std::uint32_t, 288>& lit_freq,
                      const std::array<std::uint32_t, 32>& dist_freq)
    -> std::size_t {
    std::size_t bits = 3;  // BFINAL + BTYPE=01
    for (int s = 0; s <= 143; ++s) {
        bits += static_cast<std::size_t>(lit_freq[static_cast<std::size_t>(s)]) * 8;
    }
    for (int s = 144; s <= 255; ++s) {
        bits += static_cast<std::size_t>(lit_freq[static_cast<std::size_t>(s)]) * 9;
    }
    for (int s = 256; s <= 279; ++s) {
        const auto f = lit_freq[static_cast<std::size_t>(s)];
        if (f > 0) {
            const int extra = (s >= 257) ? kLenExtraBySym[s - 257] : 0;
            bits += static_cast<std::size_t>(f) * static_cast<std::size_t>(7 + extra);
        }
    }
    for (int s = 280; s <= 285; ++s) {
        const auto f = lit_freq[static_cast<std::size_t>(s)];
        if (f > 0) {
            bits += static_cast<std::size_t>(f) *
                    static_cast<std::size_t>(8 + kLenExtraBySym[s - 257]);
        }
    }
    for (int d = 0; d < 30; ++d) {
        const auto f = dist_freq[static_cast<std::size_t>(d)];
        if (f > 0) {
            bits += static_cast<std::size_t>(f) *
                    static_cast<std::size_t>(5 + kDistExtraBySym[d]);
        }
    }
    return bits;
}

auto stored_block_bits(std::size_t cur_bit_pos, std::size_t raw_len)
    -> std::size_t {
    if (raw_len == 0) {
        const std::size_t b = cur_bit_pos + 3;
        const std::size_t pad = (8 - (b & 7)) & 7;
        return 3 + pad + 32;
    }
    std::size_t bits = 0;
    std::size_t pos = 0;
    while (pos < raw_len) {
        const std::size_t chunk = std::min<std::size_t>(65535, raw_len - pos);
        const std::size_t b = cur_bit_pos + bits + 3;
        const std::size_t pad = (8 - (b & 7)) & 7;
        bits += 3 + pad + 32 + chunk * 8;
        pos += chunk;
    }
    return bits;
}

// --------------------------------------------------------------------------
// Optimal (shortest-path) LZ77 parse with BT4 and iterative Huffman feedback.
// --------------------------------------------------------------------------

// Run backward DP on `[range_begin, range_end)` using `pool`/`cand_start`
// and write `choice_len` / `choice_dist` for that range.
auto run_dp_range(const std::uint8_t* data, int range_begin, int range_end,
                  const std::vector<MatchCand>& pool,
                  const std::vector<int>& cand_start,
                  const std::array<float, 256>& lit_cost,
                  const std::array<float, 259>& len_cost,
                  const std::array<float, 32769>& dist_cost,
                  std::vector<float>& dp,
                  std::vector<std::uint16_t>& choice_len,
                  std::vector<std::uint16_t>& choice_dist) -> std::vector<Token> {
    dp[static_cast<std::size_t>(range_end)] = 0.0f;
    for (int i = range_end - 1; i >= range_begin; --i) {
        float best = lit_cost[data[i]] + dp[static_cast<std::size_t>(i + 1)];
        int bl = 1;
        int bd = 0;
        const int max_here = std::min(kMaxMatch, range_end - i);
        const int ci_begin = cand_start[static_cast<std::size_t>(i)];
        const int ci_end = cand_start[static_cast<std::size_t>(i) + 1];
        int prev_len = kMinMatch - 1;
        for (int ci = ci_begin; ci < ci_end; ++ci) {
            const MatchCand& c = pool[static_cast<std::size_t>(ci)];
            const int d = c.dist;
            const int max_l = std::min<int>(c.len, max_here);
            if (max_l <= prev_len) continue;
            const float dbits = dist_cost[static_cast<std::size_t>(d)];

            if (max_l <= 64) {
                for (int L = prev_len + 1; L <= max_l; ++L) {
                    const float cst = len_cost[static_cast<std::size_t>(L)] + dbits +
                                      dp[static_cast<std::size_t>(i + L)];
                    if (cst < best) {
                        best = cst;
                        bl = L;
                        bd = d;
                    }
                }
            } else {
                const int dense_hi = std::min(max_l, 64);
                for (int L = prev_len + 1; L <= dense_hi; ++L) {
                    const float cst = len_cost[static_cast<std::size_t>(L)] + dbits +
                                      dp[static_cast<std::size_t>(i + L)];
                    if (cst < best) {
                        best = cst;
                        bl = L;
                        bd = d;
                    }
                }
                const int tail_lo = std::max(dense_hi + 1, max_l - 12);
                const int c_start = length_to_symbol(std::max(prev_len + 1, dense_hi + 1));
                const int c_end = length_to_symbol(max_l);
                for (int sym = c_start; sym <= c_end; ++sym) {
                    const int L = std::min(kLenBucketMax[sym], max_l);
                    if (L <= prev_len || L <= dense_hi || L >= tail_lo) continue;
                    const float cst = len_cost[static_cast<std::size_t>(L)] + dbits +
                                      dp[static_cast<std::size_t>(i + L)];
                    if (cst < best) {
                        best = cst;
                        bl = L;
                        bd = d;
                    }
                }
                for (int L = std::max(prev_len + 1, tail_lo); L <= max_l; ++L) {
                    const float cst = len_cost[static_cast<std::size_t>(L)] + dbits +
                                      dp[static_cast<std::size_t>(i + L)];
                    if (cst < best) {
                        best = cst;
                        bl = L;
                        bd = d;
                    }
                }
            }

            prev_len = max_l;
            if (prev_len >= max_here) break;
        }
        dp[static_cast<std::size_t>(i)] = best;
        choice_len[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(bl);
        choice_dist[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(bd);
    }

    std::vector<Token> tokens;
    tokens.reserve(static_cast<std::size_t>((range_end - range_begin) / 2 + 8));
    for (int p = range_begin; p < range_end;) {
        const int L = choice_len[static_cast<std::size_t>(p)];
        if (L <= 1) {
            tokens.push_back(Token::make_lit(data[p]));
            ++p;
        } else {
            const int d = choice_dist[static_cast<std::size_t>(p)];
            tokens.push_back(Token::make_match(L, d));
            p += L;
        }
    }
    return tokens;
}

void build_costs_from_freqs(const std::array<std::uint32_t, 288>& lit_freq,
                            const std::array<std::uint32_t, 32>& dist_freq,
                            bool shannon,
                            std::array<float, 256>& lit_cost,
                            std::array<float, 259>& len_cost,
                            std::array<float, 32769>& dist_cost) {
    if (shannon) {
        std::uint64_t tot_lit = 0;
        for (int s = 0; s < 286; ++s) tot_lit += lit_freq[static_cast<std::size_t>(s)];
        std::uint64_t tot_dist = 0;
        for (int s = 0; s < 30; ++s) tot_dist += dist_freq[static_cast<std::size_t>(s)];
        if (tot_lit == 0) tot_lit = 1;
        if (tot_dist == 0) tot_dist = 1;
        const float log_tot_lit = std::log2(static_cast<float>(tot_lit));
        const float log_tot_dist = std::log2(static_cast<float>(tot_dist));
        auto sc = [](std::uint32_t f, float log_tot) -> float {
            if (f == 0) return 15.0f;
            float b = log_tot - std::log2(static_cast<float>(f));
            return std::clamp(b, 1.0f, 15.0f);
        };
        for (int s = 0; s < 256; ++s) {
            lit_cost[static_cast<std::size_t>(s)] = sc(lit_freq[static_cast<std::size_t>(s)], log_tot_lit);
        }
        for (int l = 3; l <= 258; ++l) {
            const auto lc = kLenTable[static_cast<std::size_t>(l)];
            len_cost[static_cast<std::size_t>(l)] =
                sc(lit_freq[static_cast<std::size_t>(lc.sym)], log_tot_lit) +
                static_cast<float>(lc.extra);
        }
        for (int d = 1; d <= 32768; ++d) {
            const auto dc = kDistTable[static_cast<std::size_t>(d)];
            dist_cost[static_cast<std::size_t>(d)] =
                sc(dist_freq[static_cast<std::size_t>(dc.sym)], log_tot_dist) +
                static_cast<float>(dc.extra);
        }
        return;
    }

    auto lit_lens = build_huff_lengths(lit_freq, 286, 15);
    std::array<std::uint32_t, 32> df{};
    for (int i = 0; i < 30; ++i) df[static_cast<std::size_t>(i)] = dist_freq[static_cast<std::size_t>(i)];
    auto dist_lens = build_huff_lengths(df, 30, 15);

    int max_lit_len = 0;
    for (int s = 0; s < 286; ++s) {
        if (lit_lens[static_cast<std::size_t>(s)] > max_lit_len) {
            max_lit_len = lit_lens[static_cast<std::size_t>(s)];
        }
    }
    int max_dist_len = 0;
    for (int s = 0; s < 30; ++s) {
        if (dist_lens[static_cast<std::size_t>(s)] > max_dist_len) {
            max_dist_len = dist_lens[static_cast<std::size_t>(s)];
        }
    }
    const float unused_lit_cost = std::min(15.0f, static_cast<float>(max_lit_len + 1));
    const float unused_dist_cost = std::min(15.0f, static_cast<float>(max_dist_len + 1));

    for (int s = 0; s < 256; ++s) {
        const int len = lit_lens[static_cast<std::size_t>(s)];
        lit_cost[static_cast<std::size_t>(s)] =
            (len == 0) ? unused_lit_cost : static_cast<float>(len);
    }
    for (int l = 3; l <= 258; ++l) {
        const auto lc = kLenTable[static_cast<std::size_t>(l)];
        const int len = lit_lens[static_cast<std::size_t>(lc.sym)];
        const float bits = (len == 0) ? unused_lit_cost : static_cast<float>(len);
        len_cost[static_cast<std::size_t>(l)] = bits + static_cast<float>(lc.extra);
    }
    for (int d = 1; d <= 32768; ++d) {
        const auto dc = kDistTable[static_cast<std::size_t>(d)];
        const int len = dist_lens[static_cast<std::size_t>(dc.sym)];
        const float bits = (len == 0) ? unused_dist_cost : static_cast<float>(len);
        dist_cost[static_cast<std::size_t>(d)] = bits + static_cast<float>(dc.extra);
    }
}

void tally_tokens(std::span<const Token> tokens,
                  std::array<std::uint32_t, 288>& lit_freq,
                  std::array<std::uint32_t, 32>& dist_freq) {
    lit_freq.fill(0);
    dist_freq.fill(0);
    for (const Token& t : tokens) {
        if (!t.is_match()) {
            lit_freq[t.lit()]++;
        } else {
            lit_freq[static_cast<std::size_t>(257 + length_to_symbol(t.len()))]++;
            dist_freq[static_cast<std::size_t>(distance_to_symbol(t.dist()))]++;
        }
    }
    lit_freq[256] = 1;
}

// Forward declaration for block splitting inside optimal tokenize.
void find_block_splits(std::span<const Token> tokens, std::size_t a,
                       std::size_t b, std::vector<std::size_t>& splits);

struct SubblockResult {
    std::size_t byte_off = 0;
    std::size_t byte_len = 0;
    std::vector<Token> tokens;
    std::array<std::uint32_t, 288> lit_freq{};
    std::array<std::uint32_t, 32> dist_freq{};
};

auto lz77_optimal_subblocks(const std::uint8_t* data, std::size_t size,
                            int level, int bt_depth, int nice_len,
                            int iterations, std::size_t hist = 0)
    -> std::vector<SubblockResult> {
    const int N = static_cast<int>(size);
    if (N <= 0) return {};

    const auto* full_data = data - hist;
    const std::size_t full_size = size + hist;

    DeflateWorkspace& ws = get_workspace();
    const std::int32_t base = ws.begin_block(full_size);

    for (std::size_t i = 0; i < hist; ++i) {
        bt_skip_insert(full_data, full_size, i, ws, base, 16, 64);
    }

    std::vector<MatchCand> pool;
    pool.reserve(static_cast<std::size_t>(N) * 2);
    std::vector<int> cand_start(static_cast<std::size_t>(N) + 1, 0);
    for (int p = 0; p < N; ++p) {
        cand_start[static_cast<std::size_t>(p)] = static_cast<int>(pool.size());
        MatchCand local[kMaxCands];
        const std::size_t up = hist + static_cast<std::size_t>(p);
        const int nc = bt_find_and_insert(full_data, full_size, up, ws, base,
                                          bt_depth, nice_len, local);
        for (int j = 0; j < nc; ++j) pool.push_back(local[j]);
    }
    cand_start[static_cast<std::size_t>(N)] = static_cast<int>(pool.size());

    // Seed symbol statistics with greedy parse over `pool`.
    std::array<std::uint32_t, 288> lit_freq{};
    std::array<std::uint32_t, 32> dist_freq{};
    int match_bytes = 0;
    for (int p = 0; p < N;) {
        const int end_ci = cand_start[static_cast<std::size_t>(p) + 1];
        const int start_ci = cand_start[static_cast<std::size_t>(p)];
        if (end_ci > start_ci) {
            const MatchCand& best = pool[static_cast<std::size_t>(end_ci - 1)];
            if (best.len == 3 && best.dist > 4096) {
                lit_freq[data[p]]++;
                ++p;
            } else {
                lit_freq[static_cast<std::size_t>(257 + length_to_symbol(best.len))]++;
                dist_freq[static_cast<std::size_t>(distance_to_symbol(best.dist))]++;
                match_bytes += best.len;
                p += best.len;
            }
        } else {
            lit_freq[data[p]]++;
            ++p;
        }
    }
    for (const auto& c : pool) {
        const int lsym = 257 + length_to_symbol(c.len);
        if (lit_freq[static_cast<std::size_t>(lsym)] == 0) lit_freq[static_cast<std::size_t>(lsym)] = 1;
        const int dsym = distance_to_symbol(c.dist);
        if (dist_freq[static_cast<std::size_t>(dsym)] == 0) dist_freq[static_cast<std::size_t>(dsym)] = 1;
    }
    lit_freq[256] = 1;

    // Skip multi-pass iterations if input is almost incompressible.
    if (match_bytes < N / 16) {
        iterations = 1;
    }

    std::vector<std::uint16_t> choice_len(static_cast<std::size_t>(N), 1);
    std::vector<std::uint16_t> choice_dist(static_cast<std::size_t>(N), 0);
    std::vector<float> dp(static_cast<std::size_t>(N) + 1, 0.0f);

    std::vector<Token> best_tokens;
    std::array<std::uint32_t, 288> best_lit_freq{};
    std::array<std::uint32_t, 32> best_dist_freq{};
    std::size_t best_bits = static_cast<std::size_t>(-1);
    std::size_t prev_bits = static_cast<std::size_t>(-1);

    std::array<float, 256> lit_cost{};
    std::array<float, 259> len_cost{};
    std::array<float, 32769> dist_cost{};
    std::array<std::uint32_t, 288> prev_lit_freq = lit_freq;
    std::array<std::uint32_t, 32> prev_dist_freq = dist_freq;

    if (iterations < 1) iterations = 1;
    for (int it = 0; it < iterations; ++it) {
        build_costs_from_freqs(lit_freq, dist_freq, /*shannon=*/(it == 0),
                               lit_cost, len_cost, dist_cost);
        auto tokens = run_dp_range(data, 0, N, pool, cand_start,
                                   lit_cost, len_cost, dist_cost,
                                   dp, choice_len, choice_dist);

        tally_tokens(tokens, lit_freq, dist_freq);
        const std::size_t cur_bits = dynamic_block_bits(lit_freq, dist_freq);
        if (cur_bits < best_bits) {
            best_bits = cur_bits;
            best_tokens = std::move(tokens);
            best_lit_freq = lit_freq;
            best_dist_freq = dist_freq;
        } else if (cur_bits == prev_bits && lit_freq == prev_lit_freq &&
                   dist_freq == prev_dist_freq) {
            break;  // converged
        } else if (it + 1 < iterations) {
            // Smooth oscillation by averaging previous and current counts.
            for (std::size_t s = 0; s < 286; ++s) {
                lit_freq[s] = (lit_freq[s] + prev_lit_freq[s] + (lit_freq[s] > 0 || prev_lit_freq[s] > 0 ? 1u : 0u)) >> 1;
            }
            lit_freq[256] = 1;
            for (std::size_t d = 0; d < 30; ++d) {
                dist_freq[d] = (dist_freq[d] + prev_dist_freq[d] + (dist_freq[d] > 0 || prev_dist_freq[d] > 0 ? 1u : 0u)) >> 1;
            }
        }
        prev_bits = cur_bits;
        prev_lit_freq = lit_freq;
        prev_dist_freq = dist_freq;
    }

    // MDL block splitting + per-subblock re-parse over existing BT4 `pool`.
    std::vector<std::size_t> bounds;
    bounds.push_back(0);
    if (best_tokens.size() >= 1024) {
        find_block_splits(best_tokens, 0, best_tokens.size(), bounds);
    }
    bounds.push_back(best_tokens.size());

    std::vector<SubblockResult> out;
    out.reserve(bounds.size() - 1);
    if (bounds.size() <= 2) {
        SubblockResult sub;
        sub.byte_off = 0;
        sub.byte_len = size;
        sub.tokens = std::move(best_tokens);
        sub.lit_freq = best_lit_freq;
        sub.dist_freq = best_dist_freq;
        out.push_back(std::move(sub));
        return out;
    }

    const int sub_passes = (level >= 11) ? 2 : 1;
    std::size_t byte_cursor = 0;
    for (std::size_t bi = 0; bi + 1 < bounds.size(); ++bi) {
        const std::size_t t0 = bounds[bi];
        const std::size_t t1 = bounds[bi + 1];
        std::vector<Token> sub_toks(best_tokens.begin() + static_cast<std::ptrdiff_t>(t0),
                                    best_tokens.begin() + static_cast<std::ptrdiff_t>(t1));
        std::array<std::uint32_t, 288> sub_lit{};
        std::array<std::uint32_t, 32> sub_dist{};
        std::size_t sub_bytes = 0;
        for (const Token& t : sub_toks) {
            if (!t.is_match()) {
                sub_lit[t.lit()]++;
                sub_bytes += 1;
            } else {
                sub_lit[static_cast<std::size_t>(257 + length_to_symbol(t.len()))]++;
                sub_dist[static_cast<std::size_t>(distance_to_symbol(t.dist()))]++;
                sub_bytes += static_cast<std::size_t>(t.len());
            }
        }
        sub_lit[256] = 1;
        std::size_t sub_best_bits = dynamic_block_bits(sub_lit, sub_dist);

        const int rb = static_cast<int>(byte_cursor);
        const int re = static_cast<int>(byte_cursor + sub_bytes);
        if (re > rb) {
            auto cur_lit = sub_lit;
            auto cur_dist = sub_dist;
            for (int sp = 0; sp < sub_passes; ++sp) {
                build_costs_from_freqs(cur_lit, cur_dist, /*shannon=*/false,
                                       lit_cost, len_cost, dist_cost);
                auto cand_toks = run_dp_range(data, rb, re, pool, cand_start,
                                              lit_cost, len_cost, dist_cost,
                                              dp, choice_len, choice_dist);
                tally_tokens(cand_toks, cur_lit, cur_dist);
                const std::size_t cand_bits = dynamic_block_bits(cur_lit, cur_dist);
                if (cand_bits < sub_best_bits) {
                    sub_best_bits = cand_bits;
                    sub_toks = std::move(cand_toks);
                    sub_lit = cur_lit;
                    sub_dist = cur_dist;
                } else {
                    break;
                }
            }
        }

        SubblockResult sub;
        sub.byte_off = byte_cursor;
        sub.byte_len = sub_bytes;
        sub.tokens = std::move(sub_toks);
        sub.lit_freq = sub_lit;
        sub.dist_freq = sub_dist;
        out.push_back(std::move(sub));
        byte_cursor += sub_bytes;
    }
    return out;
}

// --------------------------------------------------------------------------
// Block emission.
// --------------------------------------------------------------------------
void emit_stored_block(BitWriter& w, bool final,
                       std::span<const std::byte> block) {
    w.put_bits(final ? 1u : 0u, 1);  // BFINAL
    w.put_bits(0u, 2);               // BTYPE = 00 (stored)
    w.align_to_byte();
    const std::uint16_t len = static_cast<std::uint16_t>(block.size());
    const std::uint16_t nlen = static_cast<std::uint16_t>(~len);
    const std::byte hdr[4] = {
        static_cast<std::byte>(len & 0xFFu),
        static_cast<std::byte>((len >> 8) & 0xFFu),
        static_cast<std::byte>(nlen & 0xFFu),
        static_cast<std::byte>((nlen >> 8) & 0xFFu),
    };
    w.put_bytes(std::span<const std::byte>{hdr, 4});
    w.put_bytes(block);
}

void emit_stored_blocks(BitWriter& w, bool final, const std::uint8_t* data,
                        std::size_t len) {
    if (len == 0) {
        emit_stored_block(w, final, {});
        return;
    }
    std::size_t pos = 0;
    while (pos < len) {
        const std::size_t chunk = std::min<std::size_t>(65535, len - pos);
        const bool last = (pos + chunk == len);
        emit_stored_block(
            w, final && last,
            std::span<const std::byte>{
                reinterpret_cast<const std::byte*>(data + pos), chunk});
        pos += chunk;
    }
}

void emit_tokens_with_codes(BitWriter& w, std::span<const Token> tokens,
                            const HuffCodes& lit, const HuffCodes& dist) {
    for (const auto& t : tokens) {
        if (!t.is_match()) {
            const std::uint8_t c = t.lit();
            w.put_huff(lit.rev_codes[c], lit.lengths[c]);
        } else {
            const auto lc = length_code(t.len());
            w.put_huff(lit.rev_codes[static_cast<std::size_t>(lc.sym)],
                       lit.lengths[static_cast<std::size_t>(lc.sym)]);
            if (lc.extra > 0) {
                w.put_bits(static_cast<std::uint32_t>(t.len() - lc.base),
                           lc.extra);
            }
            const auto dc = distance_code(t.dist());
            w.put_huff(dist.rev_codes[static_cast<std::size_t>(dc.sym)],
                       dist.lengths[static_cast<std::size_t>(dc.sym)]);
            if (dc.extra > 0) {
                w.put_bits(static_cast<std::uint32_t>(t.dist() - dc.base),
                           dc.extra);
            }
        }
    }
    w.put_huff(lit.rev_codes[256], lit.lengths[256]);
}

void emit_fixed_block(BitWriter& w, bool final, std::span<const Token> tokens) {
    w.put_bits(final ? 1u : 0u, 1);
    w.put_bits(1u, 2);  // BTYPE = 01
    emit_tokens_with_codes(w, tokens, kFixedLitCodes, kFixedDistCodes);
}

void emit_dynamic_block_with_lengths(BitWriter& w, bool final,
                                     std::span<const Token> tokens,
                                     const std::array<int, 286>& lit_lens,
                                     const std::array<int, 30>& dist_lens) {
    w.put_bits(final ? 1u : 0u, 1);
    w.put_bits(2u, 2);  // BTYPE = 10

    const auto lit = build_canonical_codes(lit_lens, 286);
    const auto dist = build_canonical_codes(dist_lens, 30);

    int hlit = 286;
    while (hlit > 257 && lit_lens[static_cast<std::size_t>(hlit - 1)] == 0) --hlit;
    int hdist = 30;
    while (hdist > 1 && dist_lens[static_cast<std::size_t>(hdist - 1)] == 0) --hdist;
    w.put_bits(static_cast<std::uint32_t>(hlit - 257), 5);
    w.put_bits(static_cast<std::uint32_t>(hdist - 1), 5);

    std::array<std::uint32_t, 19> cl_freq{};
    std::array<ClRle, 320> rle{};
    const int rle_len = build_cl_rle(lit_lens, hlit, dist_lens, hdist, cl_freq, rle);

    const auto cl_lens = build_huff_lengths(cl_freq, 19, 7);
    const auto cl = build_canonical_codes(cl_lens, 19);

    static constexpr int order[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                      11, 4,  12, 3, 13, 2, 14, 1, 15};
    int hclen = 19;
    while (hclen > 4 && cl_lens[static_cast<std::size_t>(order[hclen - 1])] == 0) --hclen;
    w.put_bits(static_cast<std::uint32_t>(hclen - 4), 4);
    for (int i = 0; i < hclen; ++i) {
        w.put_bits(static_cast<std::uint32_t>(cl_lens[static_cast<std::size_t>(order[i])]), 3);
    }

    for (int i = 0; i < rle_len; ++i) {
        const auto& r = rle[static_cast<std::size_t>(i)];
        w.put_huff(cl.rev_codes[static_cast<std::size_t>(r.code)],
                   cl.lengths[static_cast<std::size_t>(r.code)]);
        if (r.code == 16) {
            w.put_bits(static_cast<std::uint32_t>(r.xbits), 2);
        } else if (r.code == 17) {
            w.put_bits(static_cast<std::uint32_t>(r.xbits), 3);
        } else if (r.code == 18) {
            w.put_bits(static_cast<std::uint32_t>(r.xbits), 7);
        }
    }

    emit_tokens_with_codes(w, tokens, lit, dist);
}

// Compute exact bit lengths for dynamic (with RLE optimization at level >= 9),
// fixed, and stored modes in O(alphabet) time and emit ONLY the smallest one.
void emit_best_subblock(BitWriter& w, bool final,
                        std::span<const Token> tokens,
                        const std::array<std::uint32_t, 288>& lit_freq,
                        const std::array<std::uint32_t, 32>& dist_freq,
                        const std::uint8_t* raw_data, std::size_t raw_len,
                        int level) {
    std::array<int, 286> lit_lens{};
    std::array<int, 30> dist_lens{};
    build_dynamic_lengths(lit_freq, dist_freq, level >= 9, lit_lens, dist_lens);
    const std::size_t dyn_bits =
        eval_dynamic_bits(lit_freq, dist_freq, lit_lens, dist_lens);
    const std::size_t fix_bits = fixed_block_bits(lit_freq, dist_freq);
    const std::size_t sto_bits = stored_block_bits(w.total_bits(), raw_len);

    if (sto_bits <= dyn_bits && sto_bits <= fix_bits) {
        emit_stored_blocks(w, final, raw_data, raw_len);
    } else if (dyn_bits <= fix_bits) {
        emit_dynamic_block_with_lengths(w, final, tokens, lit_lens, dist_lens);
    } else {
        emit_fixed_block(w, final, tokens);
    }
}

// Recursively partition `tokens[a..b)` into variable-sized DEFLATE blocks when
// separate dynamic Huffman trees reduce total bits, with 3-step local boundary
// refinement around the best coarse split point.
void find_block_splits(std::span<const Token> tokens, std::size_t a,
                       std::size_t b, std::vector<std::size_t>& splits) {
    constexpr std::size_t kMinSplitTokens = 512;
    constexpr int kChunks = 16;
    const std::size_t n = b - a;
    if (n < kMinSplitTokens * 2) return;

    std::array<std::array<std::uint32_t, 288>, kChunks + 1> pref_lit{};
    std::array<std::array<std::uint32_t, 32>, kChunks + 1> pref_dist{};

    for (int k = 0; k < kChunks; ++k) {
        pref_lit[static_cast<std::size_t>(k + 1)] = pref_lit[static_cast<std::size_t>(k)];
        pref_dist[static_cast<std::size_t>(k + 1)] = pref_dist[static_cast<std::size_t>(k)];
        const std::size_t start = a + (n * static_cast<std::size_t>(k)) / kChunks;
        const std::size_t end = a + (n * static_cast<std::size_t>(k + 1)) / kChunks;
        auto& lf = pref_lit[static_cast<std::size_t>(k + 1)];
        auto& df = pref_dist[static_cast<std::size_t>(k + 1)];
        for (std::size_t i = start; i < end; ++i) {
            const Token& t = tokens[i];
            if (!t.is_match()) {
                lf[t.lit()]++;
            } else {
                lf[static_cast<std::size_t>(257 + length_to_symbol(t.len()))]++;
                df[static_cast<std::size_t>(distance_to_symbol(t.dist()))]++;
            }
        }
    }

    auto whole_lit = pref_lit[kChunks];
    whole_lit[256] = 1;
    const std::size_t whole_bits =
        dynamic_block_bits(whole_lit, pref_dist[kChunks]);

    std::size_t best_bits = whole_bits;
    int best_k = -1;
    for (int k = 1; k < kChunks; ++k) {
        const std::size_t split_pos = a + (n * static_cast<std::size_t>(k)) / kChunks;
        if (split_pos - a < kMinSplitTokens || b - split_pos < kMinSplitTokens) {
            continue;
        }
        auto left_lit = pref_lit[static_cast<std::size_t>(k)];
        left_lit[256] = 1;
        std::array<std::uint32_t, 288> right_lit{};
        std::array<std::uint32_t, 32> right_dist{};
        for (std::size_t s = 0; s < 286; ++s) {
            right_lit[s] = pref_lit[kChunks][s] - pref_lit[static_cast<std::size_t>(k)][s];
        }
        right_lit[256] = 1;
        for (std::size_t d = 0; d < 30; ++d) {
            right_dist[d] = pref_dist[kChunks][d] - pref_dist[static_cast<std::size_t>(k)][d];
        }
        const std::size_t cand_bits =
            dynamic_block_bits(left_lit, pref_dist[static_cast<std::size_t>(k)]) +
            dynamic_block_bits(right_lit, right_dist);
        if (cand_bits + 16 < best_bits) {
            best_bits = cand_bits;
            best_k = k;
        }
    }

    if (best_k <= 0) return;

    std::size_t mid = a + (n * static_cast<std::size_t>(best_k)) / kChunks;
    auto mid_lit = pref_lit[static_cast<std::size_t>(best_k)];
    auto mid_dist = pref_dist[static_cast<std::size_t>(best_k)];

    // 3-step local refinement around `mid` using incremental frequency updates.
    auto eval_shifted = [&](std::size_t cand,
                            std::array<std::uint32_t, 288>& l_lit,
                            std::array<std::uint32_t, 32>& l_dist) -> std::size_t {
        l_lit = mid_lit;
        l_dist = mid_dist;
        if (cand > mid) {
            for (std::size_t i = mid; i < cand; ++i) {
                const Token& t = tokens[i];
                if (!t.is_match()) l_lit[t.lit()]++;
                else {
                    l_lit[static_cast<std::size_t>(257 + length_to_symbol(t.len()))]++;
                    l_dist[static_cast<std::size_t>(distance_to_symbol(t.dist()))]++;
                }
            }
        } else {
            for (std::size_t i = cand; i < mid; ++i) {
                const Token& t = tokens[i];
                if (!t.is_match()) l_lit[t.lit()]--;
                else {
                    l_lit[static_cast<std::size_t>(257 + length_to_symbol(t.len()))]--;
                    l_dist[static_cast<std::size_t>(distance_to_symbol(t.dist()))]--;
                }
            }
        }
        auto l_eval = l_lit;
        l_eval[256] = 1;
        std::array<std::uint32_t, 288> r_lit{};
        std::array<std::uint32_t, 32> r_dist{};
        for (std::size_t s = 0; s < 286; ++s) r_lit[s] = pref_lit[kChunks][s] - l_lit[s];
        r_lit[256] = 1;
        for (std::size_t d = 0; d < 30; ++d) r_dist[d] = pref_dist[kChunks][d] - l_dist[d];
        return dynamic_block_bits(l_eval, l_dist) + dynamic_block_bits(r_lit, r_dist);
    };

    std::size_t step = std::max<std::size_t>(128, n / 32);
    for (int iter = 0; iter < 3 && step >= 64; ++iter, step >>= 1) {
        std::array<std::uint32_t, 288> best_cand_lit{};
        std::array<std::uint32_t, 32> best_cand_dist{};
        std::size_t next_mid = mid;
        std::size_t next_bits = best_bits;

        for (int dir : {-1, 1}) {
            if (dir < 0 && mid < a + kMinSplitTokens + step) continue;
            if (dir > 0 && mid + step + kMinSplitTokens > b) continue;
            const std::size_t cand = (dir < 0) ? (mid - step) : (mid + step);
            std::array<std::uint32_t, 288> cl{};
            std::array<std::uint32_t, 32> cd{};
            const std::size_t cb = eval_shifted(cand, cl, cd);
            if (cb < next_bits) {
                next_bits = cb;
                next_mid = cand;
                best_cand_lit = cl;
                best_cand_dist = cd;
            }
        }
        if (next_mid != mid) {
            mid = next_mid;
            best_bits = next_bits;
            mid_lit = best_cand_lit;
            mid_dist = best_cand_dist;
        }
    }

    find_block_splits(tokens, a, mid, splits);
    splits.push_back(mid);
    find_block_splits(tokens, mid, b, splits);
}

constexpr std::size_t kBlockSize = 256 * 1024;  // 256 KiB per wave block

}  // namespace

auto deflate_compress(std::span<const std::byte> data, int level)
    -> std::vector<std::byte> {
    if (data.empty()) {
        BitWriter w;
        w.put_bits(1u, 1);  // BFINAL
        w.put_bits(1u, 2);  // BTYPE = fixed
        w.put_huff(0, 7);   // Fixed EOB code 256
        w.align_to_byte();
        return w.take_data();
    }
    if (level < 1) level = 1;
    if (level > 12) level = 12;

    const bool use_optimal = (level >= 9);
    int bt_depth = 48;
    int nice_len = 128;
    int opt_iterations = 3;
    switch (level) {
        case 9:  bt_depth = 48;  nice_len = 128; opt_iterations = 3; break;
        case 10: bt_depth = 96;  nice_len = 192; opt_iterations = 4; break;
        case 11: bt_depth = 192; nice_len = 258; opt_iterations = 5; break;
        case 12: bt_depth = 512; nice_len = 258; opt_iterations = 7; break;
        default: break;
    }

    const auto* p = reinterpret_cast<const std::uint8_t*>(data.data());
    const std::size_t size = data.size();

    BitWriter w;
    w.reserve(size / 2 + 64);

    struct BlockJob {
        std::size_t off = 0;
        std::size_t len = 0;
        bool final = false;
        std::vector<SubblockResult> subblocks;
    };

    const unsigned hw = detail::effective_threads();
    const std::size_t wave_size = std::min<std::size_t>(hw, 8);

    std::size_t off = 0;
    while (off < size) {
        std::vector<BlockJob> wave;
        for (std::size_t i = 0; i < wave_size && off < size; ++i) {
            const std::size_t end = std::min(off + kBlockSize, size);
            BlockJob job;
            job.off = off;
            job.len = end - off;
            job.final = (end == size);
            wave.push_back(std::move(job));
            off = end;
        }

        auto tokenize = [&](BlockJob& job) {
            const std::size_t hist = std::min<std::size_t>(kWindow, job.off);
            if (use_optimal) {
                job.subblocks = lz77_optimal_subblocks(
                    p + job.off, job.len, level, bt_depth, nice_len,
                    opt_iterations, hist);
                return;
            }

            std::array<std::uint32_t, 288> lf{};
            std::array<std::uint32_t, 32> df{};
            auto tokens = lz77_encode(p + job.off, job.len, level, lf, df, hist);

            std::vector<std::size_t> bounds;
            bounds.push_back(0);
            if (level >= 6 && tokens.size() >= 1024) {
                find_block_splits(tokens, 0, tokens.size(), bounds);
            }
            bounds.push_back(tokens.size());

            if (bounds.size() <= 2) {
                SubblockResult sub;
                sub.byte_off = 0;
                sub.byte_len = job.len;
                sub.tokens = std::move(tokens);
                sub.lit_freq = lf;
                sub.dist_freq = df;
                job.subblocks.push_back(std::move(sub));
            } else {
                std::size_t byte_cursor = 0;
                for (std::size_t bi = 0; bi + 1 < bounds.size(); ++bi) {
                    const std::size_t t0 = bounds[bi];
                    const std::size_t t1 = bounds[bi + 1];
                    SubblockResult sub;
                    sub.byte_off = byte_cursor;
                    sub.tokens.assign(tokens.begin() + static_cast<std::ptrdiff_t>(t0),
                                      tokens.begin() + static_cast<std::ptrdiff_t>(t1));
                    std::size_t sub_bytes = 0;
                    for (const Token& t : sub.tokens) {
                        if (!t.is_match()) {
                            sub.lit_freq[t.lit()]++;
                            sub_bytes += 1;
                        } else {
                            sub.lit_freq[static_cast<std::size_t>(257 + length_to_symbol(t.len()))]++;
                            sub.dist_freq[static_cast<std::size_t>(distance_to_symbol(t.dist()))]++;
                            sub_bytes += static_cast<std::size_t>(t.len());
                        }
                    }
                    sub.lit_freq[256] = 1;
                    sub.byte_len = sub_bytes;
                    byte_cursor += sub_bytes;
                    job.subblocks.push_back(std::move(sub));
                }
            }
        };

        if (wave.size() < 2 || hw <= 1) {
            for (BlockJob& job : wave) tokenize(job);
        } else {
            const unsigned nw =
                std::min<unsigned>(hw, static_cast<unsigned>(wave.size()));
            std::atomic<std::size_t> next{0};
            std::vector<std::thread> workers;
            workers.reserve(nw);
            for (unsigned t = 0; t < nw; ++t) {
                workers.emplace_back([&]() {
                    for (;;) {
                        std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                        if (i >= wave.size()) break;
                        tokenize(wave[i]);
                    }
                });
            }
            for (auto& worker : workers) worker.join();
        }

        for (const BlockJob& job : wave) {
            const auto* block_data = p + job.off;
            for (std::size_t si = 0; si < job.subblocks.size(); ++si) {
                const auto& sub = job.subblocks[si];
                const bool sub_final = job.final && (si + 1 == job.subblocks.size());
                emit_best_subblock(w, sub_final, sub.tokens, sub.lit_freq,
                                   sub.dist_freq, block_data + sub.byte_off,
                                   sub.byte_len, level);
            }
        }
    }

    w.align_to_byte();
    auto out = w.take_data();
    if (out.size() >= size) return {};
    return out;
}

// --------------------------------------------------------------------------
// Inflate (64-bit BitReader + 9-bit Huffman LUT + chunked match copy).
// --------------------------------------------------------------------------
class BitReader {
  public:
    explicit BitReader(std::span<const std::byte> d) : d_(d) {}

    void refill() {
        while (avail_ <= 56 && pos_ < d_.size()) {
            bits_ |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(d_[pos_++])) << avail_;
            avail_ += 8;
        }
    }

    auto get_bits(int n) -> std::uint32_t {
        if (n == 0) return 0;
        if (avail_ < n) {
            refill();
            if (avail_ < n) throw std::runtime_error("DEFLATE EOF");
        }
        const std::uint32_t v = static_cast<std::uint32_t>(bits_ & ((1ULL << n) - 1ULL));
        bits_ >>= n;
        avail_ -= n;
        return v;
    }

    auto get_huff(const HuffDecoder& d) -> int {
        if (avail_ < 15) refill();
        if (avail_ == 0) throw std::runtime_error("DEFLATE EOF");

        const std::uint16_t entry = d.fast_lut[static_cast<std::size_t>(bits_ & 511ULL)];
        if (entry != 0) {
            const int len = entry >> 12;
            if (len <= avail_) {
                bits_ >>= len;
                avail_ -= len;
                return entry & 0x0FFFu;
            }
        }

        int code = 0;
        for (int len = 1; len <= d.max_len; ++len) {
            code = (code << 1) | static_cast<int>(get_bits(1));
            const int idx = code - d.first_code[static_cast<std::size_t>(len)];
            if (idx >= 0 && idx < d.count[static_cast<std::size_t>(len)]) {
                return d.symbols[static_cast<std::size_t>(
                    d.first_index[static_cast<std::size_t>(len)] + idx)];
            }
        }
        throw std::runtime_error("invalid Huffman code");
    }

    void align_byte() {
        const int drop = avail_ & 7;
        bits_ >>= drop;
        avail_ -= drop;
    }

    auto get_byte() -> std::uint8_t {
        if (avail_ >= 8) {
            const std::uint8_t b = static_cast<std::uint8_t>(bits_ & 0xFFu);
            bits_ >>= 8;
            avail_ -= 8;
            return b;
        }
        if (pos_ >= d_.size()) throw std::runtime_error("DEFLATE EOF");
        return static_cast<std::uint8_t>(d_[pos_++]);
    }

    void read_bytes(std::vector<std::byte>& out, std::size_t len) {
        align_byte();
        while (len > 0 && avail_ >= 8) {
            out.push_back(static_cast<std::byte>(bits_ & 0xFFu));
            bits_ >>= 8;
            avail_ -= 8;
            --len;
        }
        if (pos_ + len > d_.size()) throw std::runtime_error("DEFLATE EOF");
        if (len > 0) {
            out.insert(out.end(), d_.begin() + static_cast<std::ptrdiff_t>(pos_),
                       d_.begin() + static_cast<std::ptrdiff_t>(pos_ + len));
            pos_ += len;
        }
    }

  private:
    std::span<const std::byte> d_;
    std::size_t pos_ = 0;
    std::uint64_t bits_ = 0;
    int avail_ = 0;
};

inline void copy_match(std::vector<std::byte>& out, std::size_t dist, std::size_t len) {
    if (dist == 0 || dist > out.size()) {
        throw std::runtime_error("invalid DEFLATE distance");
    }
    const std::size_t old_size = out.size();
    out.resize(old_size + len);
    std::byte* dst = out.data() + old_size;
    if (dist >= len) {
        std::memcpy(dst, dst - dist, len);
    } else if (dist == 1) {
        std::memset(dst, static_cast<int>(static_cast<unsigned char>(*(dst - 1))), len);
    } else {
        std::size_t copied = dist;
        std::memcpy(dst, dst - dist, dist);
        while (copied < len) {
            const std::size_t chunk = std::min(copied, len - copied);
            std::memcpy(dst + copied, dst, chunk);
            copied += chunk;
        }
    }
}

void inflate_symbols(BitReader& r, const HuffDecoder& lit,
                     const HuffDecoder& dist, std::vector<std::byte>& out) {
    static constexpr int len_base[] = {3,4,5,6,7,8,9,10,11,13,15,17,
        19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
    static constexpr int len_extra[] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,
        2,3,3,3,3,4,4,4,4,5,5,5,5,0};
    static constexpr int dist_base[] = {1,2,3,4,5,7,9,13,17,25,33,49,
        65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,
        6145,8193,12289,16385,24577};
    static constexpr int dist_extra[] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,
        6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

    while (true) {
        const int sym = r.get_huff(lit);
        if (sym == 256) break;
        if (sym < 256) {
            out.push_back(static_cast<std::byte>(sym));
        } else {
            const int idx = sym - 257;
            const int len = len_base[idx] +
                            (len_extra[idx] ? static_cast<int>(r.get_bits(len_extra[idx])) : 0);
            const int dsym = r.get_huff(dist);
            const int d = dist_base[dsym] +
                          (dist_extra[dsym] ? static_cast<int>(r.get_bits(dist_extra[dsym])) : 0);
            copy_match(out, static_cast<std::size_t>(d), static_cast<std::size_t>(len));
        }
    }
}

auto build_fixed_lit_decoder() -> const HuffDecoder& {
    static const HuffDecoder dec = [] {
        std::array<int, 288> lit_lens{};
        for (int s = 0; s <= 143; ++s) lit_lens[static_cast<std::size_t>(s)] = 8;
        for (int s = 144; s <= 255; ++s) lit_lens[static_cast<std::size_t>(s)] = 9;
        for (int s = 256; s <= 279; ++s) lit_lens[static_cast<std::size_t>(s)] = 7;
        for (int s = 280; s <= 287; ++s) lit_lens[static_cast<std::size_t>(s)] = 8;
        return build_huff_decoder(lit_lens, 288);
    }();
    return dec;
}

auto build_fixed_dist_decoder() -> const HuffDecoder& {
    static const HuffDecoder dec = [] {
        std::array<int, 32> dlens{};
        dlens.fill(5);
        return build_huff_decoder(dlens, 30);
    }();
    return dec;
}

auto inflate_block(BitReader& r, std::vector<std::byte>& out) -> bool {
    const std::uint32_t bfinal = r.get_bits(1);
    const std::uint32_t btype = r.get_bits(2);
    if (btype == 0) {
        r.align_byte();
        const std::uint8_t lo = r.get_byte(), hi = r.get_byte();
        const int len = lo | (hi << 8);
        r.get_byte(); r.get_byte();  // nlen
        r.read_bytes(out, static_cast<std::size_t>(len));
    } else if (btype == 1) {
        inflate_symbols(r, build_fixed_lit_decoder(), build_fixed_dist_decoder(), out);
    } else if (btype == 2) {
        const int hlit = static_cast<int>(r.get_bits(5)) + 257;
        const int hdist = static_cast<int>(r.get_bits(5)) + 1;
        const int hclen = static_cast<int>(r.get_bits(4)) + 4;
        static constexpr int order[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
        std::array<int, 19> cl_lens{};
        for (int i = 0; i < hclen; ++i) {
            cl_lens[static_cast<std::size_t>(order[i])] = static_cast<int>(r.get_bits(3));
        }
        const auto cl = build_huff_decoder(cl_lens, 19);
        int all_lens[320]{};
        int count = 0;
        const int total = hlit + hdist;
        while (count < total) {
            const int sym = r.get_huff(cl);
            if (sym < 16) {
                all_lens[count++] = sym;
            } else if (sym == 16) {
                const int rep = static_cast<int>(r.get_bits(2)) + 3;
                const int v = all_lens[count - 1];
                for (int i = 0; i < rep && count < total; ++i) all_lens[count++] = v;
            } else if (sym == 17) {
                const int rep = static_cast<int>(r.get_bits(3)) + 3;
                for (int i = 0; i < rep && count < total; ++i) all_lens[count++] = 0;
            } else {
                const int rep = static_cast<int>(r.get_bits(7)) + 11;
                for (int i = 0; i < rep && count < total; ++i) all_lens[count++] = 0;
            }
        }
        std::array<int, 288> lit_lens{};
        for (int i = 0; i < hlit; ++i) lit_lens[static_cast<std::size_t>(i)] = all_lens[i];
        std::array<int, 32> dist_lens{};
        for (int i = 0; i < hdist; ++i) dist_lens[static_cast<std::size_t>(i)] = all_lens[hlit + i];
        const auto lit = build_huff_decoder(lit_lens, hlit);
        const auto dist = build_huff_decoder(dist_lens, hdist);
        inflate_symbols(r, lit, dist, out);
    } else {
        throw std::runtime_error("invalid DEFLATE block type 3");
    }
    return bfinal != 0;
}

auto deflate_decompress(std::span<const std::byte> data, std::size_t expected)
    -> std::vector<std::byte> {
    BitReader r{data};
    std::vector<std::byte> out;
    if (expected > 0) out.reserve(expected);
    bool final = false;
    while (!final) {
        final = inflate_block(r, out);
    }
    return out;
}

}  // namespace fzip
