// fzip — FZAC (fzip Binary Arithmetic Codec) encoder and decoder.
#include "fzac.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include "codec.hpp"
#include "file_type.hpp"
#include "fzac_cm.hpp"

namespace fzip {

namespace {

// ---------------------------------------------------------------------------
// Constants & State Machine
// ---------------------------------------------------------------------------
constexpr int kBitModelTotalBits = 12;
constexpr std::uint32_t kBitModelTotal = 1u << kBitModelTotalBits;  // 4096
constexpr std::uint16_t kProbInit = kBitModelTotal / 2;           // 2048
constexpr std::uint32_t kTopValue = 1u << 24;

constexpr int kNumStates = 12;
constexpr int kMaxPosBits = 2;
constexpr int kNumPosStatesMax = 16;  // ((pos & pos_mask) << 2) | prev_byte_class (0..15)
constexpr int kMinRepLen = 2;
constexpr int kMinMatchLen = 3;
constexpr int kMaxMatchLen = 273;
constexpr int kNumLenSymbols = kMaxMatchLen - kMinRepLen + 1;  // 272
constexpr int kNumLenStates = 8;

constexpr std::uint32_t kNullPos = 0xFFFFFFFFu;
constexpr std::uint64_t kInfinity = std::numeric_limits<std::uint64_t>::max() / 4;
constexpr std::size_t kOptChunk = 2048;

constexpr std::uint8_t kMagic0 = 0x46;  // 'F'
constexpr std::uint8_t kMagic1 = 0x5A;  // 'Z'
constexpr std::uint8_t kVersionSingle = 1;
constexpr std::uint8_t kVersionMultiBlock = 2;
constexpr std::uint8_t kVersionCM = 3;

constexpr std::array<std::uint8_t, kNumStates> kNextStateLit = {
    0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 4, 5
};

inline constexpr auto state_after_match(std::uint8_t s) -> std::uint8_t {
    return (s < 7) ? 7 : 10;
}
inline constexpr auto state_after_rep(std::uint8_t s) -> std::uint8_t {
    return (s < 7) ? 8 : 11;
}
inline constexpr auto state_after_shortrep(std::uint8_t s) -> std::uint8_t {
    return (s < 7) ? 9 : 11;
}

// ---------------------------------------------------------------------------
// Fast Unaligned Loads & Hash Helpers
// ---------------------------------------------------------------------------
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

inline auto load_u64(const std::uint8_t* p) -> std::uint64_t {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

inline void store_u32(std::uint8_t* p, std::uint32_t v) {
    std::memcpy(p, &v, 4);
}

inline auto match_len_fast(const std::uint8_t* a, const std::uint8_t* b,
                           std::uint32_t start_l, std::uint32_t limit)
    -> std::uint32_t {
    std::uint32_t l = start_l;
    while (l + 8 <= limit) {
        std::uint64_t va, vb;
        std::memcpy(&va, a + l, 8);
        std::memcpy(&vb, b + l, 8);
        if (va != vb) {
            if constexpr (std::endian::native == std::endian::little) {
                return l + static_cast<std::uint32_t>(std::countr_zero(va ^ vb) >> 3);
            } else {
                return l + static_cast<std::uint32_t>(std::countl_zero(va ^ vb) >> 3);
            }
        }
        l += 8;
    }
    while (l < limit && a[l] == b[l]) ++l;
    return l;
}

inline auto hash3(const std::uint8_t* p) -> std::uint32_t {
    const std::uint32_t h = static_cast<std::uint32_t>(p[0]) |
                            (static_cast<std::uint32_t>(p[1]) << 8) |
                            (static_cast<std::uint32_t>(p[2]) << 16);
    return (h * 2654435761u) >> 14;  // 18-bit (262144 buckets)
}

inline auto hash4(const std::uint8_t* p, std::uint32_t mask) -> std::uint32_t {
    return (load_u32(p) * 2654435761u) & mask;
}

inline auto hash6(const std::uint8_t* p, std::uint32_t mask) -> std::uint32_t {
    const std::uint64_t v = static_cast<std::uint64_t>(load_u32(p)) |
                            (static_cast<std::uint64_t>(load_u16(p + 4)) << 32);
    const std::uint64_t h = v * 0xCF1BBCDCB7A56463ULL;
    return static_cast<std::uint32_t>((h >> 40) ^ (h >> 16)) & mask;
}

inline auto hash8(const std::uint8_t* p, std::uint32_t mask) -> std::uint32_t {
    const std::uint64_t h = load_u64(p) * 0x9E3779B97F4A7C15ULL;
    return static_cast<std::uint32_t>((h >> 40) ^ (h >> 16)) & mask;
}

inline auto len_to_dist_state(std::uint32_t len, bool use_len8) -> std::uint32_t {
    const std::uint32_t s = len - kMinMatchLen;
    if (!use_len8) return std::min<std::uint32_t>(s, 3u);
    if (s < 5u) return s;
    if (len <= 11u) return 5u;
    if (len <= 23u) return 6u;
    return 7u;
}

inline auto prev_byte_class(std::uint8_t b) -> std::uint32_t {
    if (b <= 32u) return 0u;
    if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || b == '_') return 1u;
    if (b >= '0' && b <= '9') return 2u;
    return 3u;
}

inline auto match_ctx_of(std::uint32_t pos, std::uint32_t pos_mask,
                         std::uint8_t prev_b) -> std::uint32_t {
    return ((pos & pos_mask) << 2) | prev_byte_class(prev_b);
}

inline auto rep_ctx_of(std::uint32_t pos, std::uint32_t pos_mask,
                       std::uint8_t prev_b) -> std::uint32_t {
    return (pos_mask == 0u) ? prev_byte_class(prev_b) : (pos & pos_mask);
}

// ---------------------------------------------------------------------------
// Bit-Price Lookup Table (1 bit = 64 units)
// ---------------------------------------------------------------------------
auto make_bit_prices() -> std::array<std::uint16_t, kBitModelTotal> {
    std::array<std::uint16_t, kBitModelTotal> t{};
    t[0] = 768;  // unused fallback
    for (std::uint32_t p = 1; p < kBitModelTotal; ++p) {
        const double bits = -std::log2(static_cast<double>(p) /
                                       static_cast<double>(kBitModelTotal));
        const long long scaled = std::llround(bits * 64.0);
        t[p] = static_cast<std::uint16_t>(std::max<long long>(1, scaled));
    }
    return t;
}

inline auto bit_prices() -> const std::array<std::uint16_t, kBitModelTotal>& {
    static const auto kTable = make_bit_prices();
    return kTable;
}

inline auto price_bit(std::uint32_t prob, int bit) -> std::uint32_t {
    const std::uint32_t idx = bit ? (kBitModelTotal - prob) : prob;
    return bit_prices()[idx];
}

inline auto price_bit_mixed(std::uint16_t pa, std::uint16_t pb, int bit)
    -> std::uint32_t {
    const std::uint32_t p = (static_cast<std::uint32_t>(pa) +
                             static_cast<std::uint32_t>(pb)) >> 1;
    return price_bit(p, bit);
}

template <int Shift>
inline void update_prob(std::uint16_t& prob, int bit) {
    if (bit == 0) {
        prob = static_cast<std::uint16_t>(prob + ((kBitModelTotal - prob) >> Shift));
    } else {
        prob = static_cast<std::uint16_t>(prob - (prob >> Shift));
    }
}

inline auto mix2_backoff(std::uint16_t p0, std::uint16_t p1) -> std::uint32_t {
    const std::uint32_t e1 = (p1 == kProbInit) ? p0 : p1;
    return (static_cast<std::uint32_t>(p0) + e1) >> 1;
}

inline auto mix3_backoff(std::uint16_t p0, std::uint16_t p1, std::uint16_t p2)
    -> std::uint32_t {
    const std::uint32_t e1 = (p1 == kProbInit) ? p0 : p1;
    const std::uint32_t e2 = (p2 == kProbInit) ? e1 : p2;
    return (static_cast<std::uint32_t>(p0) + (e1 << 1) + e2) >> 2;
}

inline auto mix4_backoff(std::uint16_t p0, std::uint16_t p1,
                         std::uint16_t p2, std::uint16_t p3) -> std::uint32_t {
    const std::uint32_t e1 = (p1 == kProbInit) ? p0 : p1;
    const std::uint32_t e2 = (p2 == kProbInit) ? e1 : p2;
    const std::uint32_t e3 = (p3 == kProbInit) ? e2 : p3;
    return (static_cast<std::uint32_t>(p0) + (e1 << 1) + (e2 << 1) + e3 * 3u) >> 3;
}

inline auto mix5_backoff(std::uint16_t p0, std::uint16_t p1,
                         std::uint16_t p2, std::uint16_t p3,
                         std::uint16_t p4) -> std::uint32_t {
    const std::uint32_t e1 = (p1 == kProbInit) ? p0 : p1;
    const std::uint32_t e2 = (p2 == kProbInit) ? e1 : p2;
    const std::uint32_t e3 = (p3 == kProbInit) ? e2 : p3;
    const std::uint32_t e4 = (p4 == kProbInit) ? e3 : p4;
    return (static_cast<std::uint32_t>(p0) + (e1 << 1) + e2 * 3u +
            e3 * 5u + e4 * 5u + 8u) >> 4;
}

inline auto scale_prob(std::uint32_t p_raw, std::int32_t w_adapt) -> std::uint32_t {
    const std::int32_t centered = static_cast<std::int32_t>(p_raw) - 2048;
    const std::int32_t scaled = 2048 + ((centered * w_adapt) >> 8);
    return static_cast<std::uint32_t>(std::clamp(scaled, 1, 4095));
}

// ---------------------------------------------------------------------------
// Dual-Rate Probability Pair
// ---------------------------------------------------------------------------
struct DualProb {
    std::uint16_t fast = kProbInit;
    std::uint16_t slow = kProbInit;

    void reset() {
        fast = kProbInit;
        slow = kProbInit;
    }

    auto mix() const -> std::uint32_t {
        return (static_cast<std::uint32_t>(fast) +
                static_cast<std::uint32_t>(slow)) >> 1;
    }

    auto price(int bit) const -> std::uint32_t {
        return price_bit(mix(), bit);
    }

    void update(int bit) {
        if (bit == 0) {
            fast = static_cast<std::uint16_t>(
                fast + ((kBitModelTotal - fast) >> 4));
            slow = static_cast<std::uint16_t>(
                slow + ((kBitModelTotal - slow) >> 6));
        } else {
            fast = static_cast<std::uint16_t>(fast - (fast >> 4));
            slow = static_cast<std::uint16_t>(slow - (slow >> 6));
        }
    }
};

// ---------------------------------------------------------------------------
// Carryless / 64-bit Low Range Encoder & Range Decoder
// ---------------------------------------------------------------------------
class RangeEncoder {
  public:
    void reserve(std::size_t bytes) { out_.reserve(bytes); }

    void encode_bit(std::uint16_t& prob, int bit) {
        const std::uint32_t p = prob;
        const std::uint32_t bound = (range_ >> kBitModelTotalBits) * p;
        if (bit == 0) {
            range_ = bound;
            prob = static_cast<std::uint16_t>(p + ((kBitModelTotal - p) >> 5));
        } else {
            low_ += bound;
            range_ -= bound;
            prob = static_cast<std::uint16_t>(p - (p >> 5));
        }
        while (range_ < kTopValue) {
            range_ <<= 8;
            shift_low();
        }
    }

    template <int ShiftA, int ShiftB>
    void encode_bit_mixed(std::uint16_t& pa, std::uint16_t& pb, int bit) {
        const std::uint32_t p = (static_cast<std::uint32_t>(pa) +
                                 static_cast<std::uint32_t>(pb)) >> 1;
        const std::uint32_t bound = (range_ >> kBitModelTotalBits) * p;
        if (bit == 0) {
            range_ = bound;
            pa = static_cast<std::uint16_t>(pa + ((kBitModelTotal - pa) >> ShiftA));
            pb = static_cast<std::uint16_t>(pb + ((kBitModelTotal - pb) >> ShiftB));
        } else {
            low_ += bound;
            range_ -= bound;
            pa = static_cast<std::uint16_t>(pa - (pa >> ShiftA));
            pb = static_cast<std::uint16_t>(pb - (pb >> ShiftB));
        }
        while (range_ < kTopValue) {
            range_ <<= 8;
            shift_low();
        }
    }

    void encode_bit_prob(std::uint32_t p, int bit) {
        const std::uint32_t bound = (range_ >> kBitModelTotalBits) * p;
        if (bit == 0) {
            range_ = bound;
        } else {
            low_ += bound;
            range_ -= bound;
        }
        while (range_ < kTopValue) {
            range_ <<= 8;
            shift_low();
        }
    }

    void encode_dual(DualProb& dp, int bit) {
        encode_bit_mixed<4, 6>(dp.fast, dp.slow, bit);
    }

    void encode_direct_bits(std::uint32_t value, int num_bits) {
        for (int i = num_bits - 1; i >= 0; --i) {
            range_ >>= 1;
            if ((value >> i) & 1u) {
                low_ += range_;
            }
            while (range_ < kTopValue) {
                range_ <<= 8;
                shift_low();
            }
        }
    }

    void finish() {
        for (int i = 0; i < 5; ++i) {
            shift_low();
        }
    }

    auto buffer() -> std::vector<std::byte>& { return out_; }
    auto buffer() const -> const std::vector<std::byte>& { return out_; }

  private:
    void emit_byte(std::uint8_t b) {
        if (!skipped_first_) {
            skipped_first_ = true;
            return;
        }
        out_.push_back(static_cast<std::byte>(b));
    }

    void shift_low() {
        const std::uint32_t low_hi = static_cast<std::uint32_t>(low_ >> 32);
        if (low_hi != 0 || low_ < 0xFF000000ULL) {
            std::uint8_t temp = cache_;
            do {
                emit_byte(static_cast<std::uint8_t>(temp + low_hi));
                temp = 0xFFu;
            } while (--cache_size_ != 0);
            cache_ = static_cast<std::uint8_t>(static_cast<std::uint32_t>(low_) >> 24);
        }
        ++cache_size_;
        low_ = (low_ & 0x00FFFFFFULL) << 8;
    }

    std::vector<std::byte> out_;
    std::uint64_t low_ = 0;
    std::uint32_t range_ = 0xFFFFFFFFu;
    std::uint64_t cache_size_ = 1;
    std::uint8_t cache_ = 0;
    bool skipped_first_ = false;
};

class RangeDecoder {
  public:
    RangeDecoder(const std::byte* begin, const std::byte* end)
        : pos_(reinterpret_cast<const std::uint8_t*>(begin)),
          end_(reinterpret_cast<const std::uint8_t*>(end)) {
        for (int i = 0; i < 4; ++i) {
            code_ = (code_ << 8) | next_byte();
        }
    }

    auto decode_bit(std::uint16_t& prob) -> int {
        const std::uint32_t p = prob;
        const std::uint32_t bound = (range_ >> kBitModelTotalBits) * p;
        int bit;
        if (code_ < bound) {
            range_ = bound;
            prob = static_cast<std::uint16_t>(p + ((kBitModelTotal - p) >> 5));
            bit = 0;
        } else {
            range_ -= bound;
            code_ -= bound;
            prob = static_cast<std::uint16_t>(p - (p >> 5));
            bit = 1;
        }
        while (range_ < kTopValue) {
            range_ <<= 8;
            code_ = (code_ << 8) | next_byte();
        }
        return bit;
    }

    template <int ShiftA, int ShiftB>
    auto decode_bit_mixed(std::uint16_t& pa, std::uint16_t& pb) -> int {
        const std::uint32_t p = (static_cast<std::uint32_t>(pa) +
                                 static_cast<std::uint32_t>(pb)) >> 1;
        const std::uint32_t bound = (range_ >> kBitModelTotalBits) * p;
        int bit;
        if (code_ < bound) {
            range_ = bound;
            pa = static_cast<std::uint16_t>(pa + ((kBitModelTotal - pa) >> ShiftA));
            pb = static_cast<std::uint16_t>(pb + ((kBitModelTotal - pb) >> ShiftB));
            bit = 0;
        } else {
            range_ -= bound;
            code_ -= bound;
            pa = static_cast<std::uint16_t>(pa - (pa >> ShiftA));
            pb = static_cast<std::uint16_t>(pb - (pb >> ShiftB));
            bit = 1;
        }
        while (range_ < kTopValue) {
            range_ <<= 8;
            code_ = (code_ << 8) | next_byte();
        }
        return bit;
    }

    auto decode_bit_prob(std::uint32_t p) -> int {
        const std::uint32_t bound = (range_ >> kBitModelTotalBits) * p;
        int bit;
        if (code_ < bound) {
            range_ = bound;
            bit = 0;
        } else {
            range_ -= bound;
            code_ -= bound;
            bit = 1;
        }
        while (range_ < kTopValue) {
            range_ <<= 8;
            code_ = (code_ << 8) | next_byte();
        }
        return bit;
    }

    auto decode_dual(DualProb& dp) -> int {
        return decode_bit_mixed<4, 6>(dp.fast, dp.slow);
    }

    auto decode_direct_bits(int num_bits) -> std::uint32_t {
        std::uint32_t res = 0;
        for (int i = 0; i < num_bits; ++i) {
            range_ >>= 1;
            const std::uint32_t bit = (code_ >= range_) ? 1u : 0u;
            code_ -= bit ? range_ : 0u;
            while (range_ < kTopValue) {
                range_ <<= 8;
                code_ = (code_ << 8) | next_byte();
            }
            res = (res << 1) | bit;
        }
        return res;
    }

  private:
    auto next_byte() -> std::uint32_t {
        return (pos_ < end_) ? static_cast<std::uint32_t>(*pos_++) : 0u;
    }

    const std::uint8_t* pos_;
    const std::uint8_t* end_;
    std::uint32_t range_ = 0xFFFFFFFFu;
    std::uint32_t code_ = 0;
};

// ---------------------------------------------------------------------------
// Bit-Tree & Reverse Bit-Tree Helpers
// ---------------------------------------------------------------------------
template <int NumBits>
inline void encode_bit_tree(RangeEncoder& rc, std::uint16_t* probs,
                            std::uint32_t sym) {
    std::uint32_t m = 1;
    for (int i = NumBits - 1; i >= 0; --i) {
        const int bit = static_cast<int>((sym >> i) & 1u);
        rc.encode_bit(probs[m], bit);
        m = (m << 1) | static_cast<std::uint32_t>(bit);
    }
}

template <int NumBits>
inline auto decode_bit_tree(RangeDecoder& rd, std::uint16_t* probs)
    -> std::uint32_t {
    std::uint32_t m = 1;
    for (int i = 0; i < NumBits; ++i) {
        m = (m << 1) | static_cast<std::uint32_t>(rd.decode_bit(probs[m]));
    }
    return m - (1u << NumBits);
}

template <int NumBits>
inline auto price_bit_tree(const std::uint16_t* probs, std::uint32_t sym)
    -> std::uint32_t {
    std::uint32_t price = 0;
    std::uint32_t m = 1;
    for (int i = NumBits - 1; i >= 0; --i) {
        const int bit = static_cast<int>((sym >> i) & 1u);
        price += price_bit(probs[m], bit);
        m = (m << 1) | static_cast<std::uint32_t>(bit);
    }
    return price;
}

inline void encode_reverse_bit_tree(RangeEncoder& rc, std::uint16_t* probs,
                                    int num_bits, std::uint32_t sym) {
    std::uint32_t m = 1;
    for (int i = 0; i < num_bits; ++i) {
        const int bit = static_cast<int>((sym >> i) & 1u);
        rc.encode_bit(probs[m], bit);
        m = (m << 1) | static_cast<std::uint32_t>(bit);
    }
}

inline auto decode_reverse_bit_tree(RangeDecoder& rd, std::uint16_t* probs,
                                    int num_bits) -> std::uint32_t {
    std::uint32_t m = 1;
    std::uint32_t sym = 0;
    for (int i = 0; i < num_bits; ++i) {
        const int bit = rd.decode_bit(probs[m]);
        m = (m << 1) | static_cast<std::uint32_t>(bit);
        sym |= (static_cast<std::uint32_t>(bit) << i);
    }
    return sym;
}

inline auto price_reverse_bit_tree(const std::uint16_t* probs, int num_bits,
                                   std::uint32_t sym) -> std::uint32_t {
    std::uint32_t price = 0;
    std::uint32_t m = 1;
    for (int i = 0; i < num_bits; ++i) {
        const int bit = static_cast<int>((sym >> i) & 1u);
        price += price_bit(probs[m], bit);
        m = (m << 1) | static_cast<std::uint32_t>(bit);
    }
    return price;
}

// ---------------------------------------------------------------------------
// Distance Slot Mapping
// ---------------------------------------------------------------------------
inline auto get_pos_slot(std::uint32_t dist) -> std::uint32_t {
    const std::uint32_t d = dist - 1u;
    if (d < 4u) return d;
    const int bit_pos = static_cast<int>(std::bit_width(d)) - 1;
    return static_cast<std::uint32_t>((bit_pos << 1) +
                                      static_cast<int>((d >> (bit_pos - 1)) & 1u));
}

inline auto min_dist_for_slot(std::uint32_t slot) -> std::uint32_t {
    if (slot < 4u) return slot + 1u;
    const int num_extra = static_cast<int>(slot >> 1) - 1;
    return ((2u | (slot & 1u)) << num_extra) + 1u;
}

// ---------------------------------------------------------------------------
// Length Coder & Model State
// ---------------------------------------------------------------------------
struct LenCoder {
    std::uint16_t choice = kProbInit;
    std::uint16_t choice2 = kProbInit;
    std::array<std::array<std::uint16_t, 8>, kNumPosStatesMax> low{};
    std::array<std::array<std::uint16_t, 8>, kNumPosStatesMax> mid{};
    std::array<std::uint16_t, 256> high{};

    void init() {
        choice = kProbInit;
        choice2 = kProbInit;
        for (auto& a : low) a.fill(kProbInit);
        for (auto& a : mid) a.fill(kProbInit);
        high.fill(kProbInit);
    }

    void encode(RangeEncoder& rc, std::uint32_t pos_state, std::uint32_t len,
                std::uint32_t min_len) {
        const std::uint32_t sym = len - min_len;
        if (sym < 8u) {
            rc.encode_bit(choice, 0);
            encode_bit_tree<3>(rc, low[pos_state].data(), sym);
        } else if (sym < 16u) {
            rc.encode_bit(choice, 1);
            rc.encode_bit(choice2, 0);
            encode_bit_tree<3>(rc, mid[pos_state].data(), sym - 8u);
        } else {
            rc.encode_bit(choice, 1);
            rc.encode_bit(choice2, 1);
            encode_bit_tree<8>(rc, high.data(), sym - 16u);
        }
    }

    auto decode(RangeDecoder& rd, std::uint32_t pos_state,
                std::uint32_t min_len) -> std::uint32_t {
        if (rd.decode_bit(choice) == 0) {
            return decode_bit_tree<3>(rd, low[pos_state].data()) + min_len;
        }
        if (rd.decode_bit(choice2) == 0) {
            return decode_bit_tree<3>(rd, mid[pos_state].data()) + 8u + min_len;
        }
        return decode_bit_tree<8>(rd, high.data()) + 16u + min_len;
    }

    auto price_direct(std::uint32_t pos_state, std::uint32_t len,
                      std::uint32_t min_len) const -> std::uint32_t {
        const std::uint32_t sym = len - min_len;
        if (sym < 8u) {
            return price_bit(choice, 0) +
                   price_bit_tree<3>(low[pos_state].data(), sym);
        }
        if (sym < 16u) {
            return price_bit(choice, 1) + price_bit(choice2, 0) +
                   price_bit_tree<3>(mid[pos_state].data(), sym - 8u);
        }
        return price_bit(choice, 1) + price_bit(choice2, 1) +
               price_bit_tree<8>(high.data(), sym - 16u);
    }
};

struct ModelState {
    std::uint8_t lit_mode = 0;
    std::uint8_t lc = 4;
    std::uint8_t pb = 0;
    bool use_o3 = false;
    bool use_o4 = false;
    bool o2_matched = false;
    bool use_len8 = false;
    std::uint32_t pos_mask = 0;
    std::uint32_t o2_stride = 256u;
    std::uint32_t o2_mask = 511u;
    std::uint32_t o3_mask = 0u;
    std::uint32_t o4_mask = 0u;

    std::array<std::array<DualProb, kNumPosStatesMax>, kNumStates> is_match{};
    std::array<DualProb, kNumStates> is_rep{};
    std::array<DualProb, kNumStates> is_rep0{};
    std::array<std::array<DualProb, kNumPosStatesMax>, kNumStates> is_rep0_long{};
    std::array<DualProb, kNumStates> is_rep1{};
    std::array<DualProb, kNumStates> is_rep2{};

    LenCoder match_len_coder;
    LenCoder rep_len_coder;

    std::array<std::uint16_t, 64> slot_o0_coder{};
    std::array<std::array<std::uint16_t, 64>, kNumLenStates> slot_coder{};
    std::array<std::uint16_t, 128> special_coder{};
    std::array<std::uint16_t, 16> align_coder{};

    std::int32_t lit_gain = 512;
    std::int32_t w_adapt = 256;

    std::array<std::uint16_t, 768> o0_probs{};
    std::vector<std::uint16_t> lit_probs;
    std::vector<std::uint16_t> order2_probs;
    std::vector<std::uint16_t> order3_probs;
    std::vector<std::uint16_t> order4_probs;

    void init(std::uint8_t mode, std::uint8_t pbits, std::uint32_t block_n = 0u,
              bool alloc_lit_tables = true) {
        lit_mode = mode;
        pb = pbits;
        pos_mask = (1u << pb) - 1u;
        if (block_n <= 524288u) {
            use_o3 = false;
            use_o4 = false;
            o2_matched = false;
            use_len8 = false;
            o2_stride = 256u;
            o3_mask = 0u;
            o4_mask = 0u;
            switch (lit_mode) {
                case 0: lc = 4; o2_mask = 511u; break;
                case 1: lc = 6; o2_mask = 1023u; break;
                case 2: lc = 7; o2_mask = 2047u; break;
                default: lc = 8; o2_mask = 4095u; break;
            }
        } else {
            lc = 8;
            o2_mask = 16383u;
            o2_matched = false;
            use_len8 = true;
            o2_stride = 256u;
            use_o3 = true;
            o3_mask = 32767u;
            use_o4 = false;
            o4_mask = 0u;
        }

        for (auto& row : is_match) {
            for (auto& d : row) d.reset();
        }
        for (auto& d : is_rep) d.reset();
        for (auto& d : is_rep0) d.reset();
        for (auto& row : is_rep0_long) {
            for (auto& d : row) d.reset();
        }
        for (auto& d : is_rep1) d.reset();
        for (auto& d : is_rep2) d.reset();

        match_len_coder.init();
        rep_len_coder.init();

        slot_o0_coder.fill(kProbInit);
        for (auto& s : slot_coder) s.fill(kProbInit);
        special_coder.fill(kProbInit);
        align_coder.fill(kProbInit);

        lit_gain = 512;
        w_adapt = 256;

        o0_probs.fill(kProbInit);
        if (alloc_lit_tables) {
            const std::size_t lit_cnt = (std::size_t{1} << lc) * 768u;
            lit_probs.assign(lit_cnt, kProbInit);
            order2_probs.assign(static_cast<std::size_t>(o2_mask + 1u) * o2_stride, kProbInit);
            if (use_o3) {
                order3_probs.assign(static_cast<std::size_t>(o3_mask + 1u) * 256u, kProbInit);
            } else {
                order3_probs.clear();
            }
            if (use_o4) {
                order4_probs.assign(static_cast<std::size_t>(o4_mask + 1u) * 256u, kProbInit);
            } else {
                order4_probs.clear();
            }
        }
    }
};

inline auto lit_ctx1(std::uint8_t prev_byte, std::uint8_t lc) -> std::uint32_t {
    return static_cast<std::uint32_t>(prev_byte) >> (8 - lc);
}

inline auto lit_ctx2(std::uint32_t pos, std::uint8_t prev_byte,
                     std::uint8_t prev2_byte, std::uint32_t pos_mask,
                     std::uint32_t o2_mask) -> std::uint32_t {
    if (pos_mask == 0u && o2_mask == 16383u &&
        ((prev_byte | prev2_byte) & 0x80u) == 0u) {
        return static_cast<std::uint32_t>(prev_byte) |
               (static_cast<std::uint32_t>(prev2_byte) << 7);
    }
    const std::uint32_t v = static_cast<std::uint32_t>(prev_byte) |
                            (static_cast<std::uint32_t>(prev2_byte) << 8) |
                            ((pos & pos_mask) << 16);
    return ((v * 2654435761u) >> 16) & o2_mask;
}

inline auto lit_ctx3(std::uint32_t pos, std::uint8_t prev_byte,
                     std::uint8_t prev2_byte, std::uint8_t prev3_byte,
                     std::uint32_t pos_mask, std::uint32_t o3_mask)
    -> std::uint32_t {
    std::uint32_t v = static_cast<std::uint32_t>(prev_byte) |
                      (static_cast<std::uint32_t>(prev2_byte) << 8) |
                      (static_cast<std::uint32_t>(prev3_byte) << 16) |
                      ((pos & pos_mask) << 24);
    v *= 0x9E3779B1u;
    v ^= v >> 15;
    v *= 0x85EBCA77u;
    return (v >> 16) & o3_mask;
}

inline auto lit_ctx4(std::uint32_t pos, std::uint8_t prev_byte,
                     std::uint8_t prev2_byte, std::uint8_t prev3_byte,
                     std::uint8_t prev4_byte, std::uint32_t pos_mask,
                     std::uint32_t o4_mask) -> std::uint32_t {
    std::uint64_t v = static_cast<std::uint64_t>(prev_byte) |
                      (static_cast<std::uint64_t>(prev2_byte) << 8) |
                      (static_cast<std::uint64_t>(prev3_byte) << 16) |
                      (static_cast<std::uint64_t>(prev4_byte) << 24) |
                      (static_cast<std::uint64_t>(pos & pos_mask) << 32);
    v *= 0x9E3779B97F4A7C15ULL;
    v ^= v >> 27;
    v *= 0xBF58476D1CE4E5B9ULL;
    return static_cast<std::uint32_t>(v >> 32) & o4_mask;
}

inline void update_lit_gain(ModelState& m, std::uint32_t raw_byte_cost) {
    m.lit_gain = std::clamp(
        m.lit_gain + (508 - static_cast<std::int32_t>(raw_byte_cost)),
        -512, 1024);
    if (m.lit_gain >= 256) {
        m.w_adapt = 256;
    } else if (m.lit_gain <= -128) {
        m.w_adapt = 16;
    } else {
        m.w_adapt = 16 + ((m.lit_gain + 128) * 240) / 384;
    }
}

inline void encode_literal(RangeEncoder& rc, ModelState& m, std::uint32_t pos,
                           std::uint8_t prev_byte, std::uint8_t prev2_byte,
                           std::uint8_t prev3_byte, std::uint8_t prev4_byte,
                           bool matched, std::uint8_t match_byte,
                           std::uint8_t symbol_byte) {
    std::uint16_t* o0_p = m.o0_probs.data();
    std::uint16_t* lit_p = m.lit_probs.data() + lit_ctx1(prev_byte, m.lc) * 768u;
    std::uint16_t* o2_p = m.order2_probs.data() +
        static_cast<std::size_t>(
            lit_ctx2(pos, prev_byte, prev2_byte, m.pos_mask, m.o2_mask)) *
            m.o2_stride;
    std::uint16_t* o3_p = m.use_o3
        ? (m.order3_probs.data() +
           lit_ctx3(pos, prev_byte, prev2_byte, prev3_byte, m.pos_mask, m.o3_mask) * 256u)
        : nullptr;
    std::uint16_t* o4_p = m.use_o4
        ? (m.order4_probs.data() +
           lit_ctx4(pos, prev_byte, prev2_byte, prev3_byte, prev4_byte, m.pos_mask, m.o4_mask) * 256u)
        : nullptr;
    const std::int32_t w = m.w_adapt;
    const bool high_noise = (m.lit_gain < 0);
    std::uint32_t raw_byte_cost = 0;
    std::uint32_t sym = 1;
    for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
        const int bit = (symbol_byte >> bit_idx) & 1;
        if (matched) {
            const int match_bit = (match_byte >> bit_idx) & 1;
            const std::uint32_t idx =
                0x100u + (static_cast<std::uint32_t>(match_bit) << 8) + sym;
            const std::uint32_t p_raw = m.o2_matched
                ? mix3_backoff(o0_p[idx], lit_p[idx], o2_p[idx])
                : mix2_backoff(o0_p[idx], lit_p[idx]);
            const std::uint32_t p_used = scale_prob(p_raw, w);
            rc.encode_bit_prob(p_used, bit);
            raw_byte_cost += price_bit(p_raw, bit);
            if (!high_noise) {
                update_prob<5>(o0_p[idx], bit);
                update_prob<4>(lit_p[idx], bit);
                if (m.o2_matched) update_prob<4>(o2_p[idx], bit);
            } else {
                update_prob<7>(o0_p[idx], bit);
                update_prob<7>(lit_p[idx], bit);
            }
            sym = (sym << 1) | static_cast<std::uint32_t>(bit);
            if (bit != match_bit) matched = false;
        } else {
            std::uint32_t p_raw;
            if (o4_p) {
                p_raw = mix5_backoff(o0_p[sym], lit_p[sym], o2_p[sym], o3_p[sym], o4_p[sym]);
            } else if (o3_p) {
                p_raw = mix4_backoff(o0_p[sym], lit_p[sym], o2_p[sym], o3_p[sym]);
            } else {
                p_raw = mix3_backoff(o0_p[sym], lit_p[sym], o2_p[sym]);
            }
            const std::uint32_t p_used = scale_prob(p_raw, w);
            rc.encode_bit_prob(p_used, bit);
            raw_byte_cost += price_bit(p_raw, bit);
            if (!high_noise) {
                update_prob<5>(o0_p[sym], bit);
                update_prob<4>(lit_p[sym], bit);
                update_prob<4>(o2_p[sym], bit);
                if (o3_p) update_prob<4>(o3_p[sym], bit);
                if (o4_p) update_prob<4>(o4_p[sym], bit);
            } else {
                update_prob<7>(o0_p[sym], bit);
                update_prob<7>(lit_p[sym], bit);
            }
            sym = (sym << 1) | static_cast<std::uint32_t>(bit);
        }
    }
    update_lit_gain(m, raw_byte_cost);
}

inline auto decode_literal(RangeDecoder& rd, ModelState& m, std::uint32_t pos,
                           std::uint8_t prev_byte, std::uint8_t prev2_byte,
                           std::uint8_t prev3_byte, std::uint8_t prev4_byte,
                           bool matched, std::uint8_t match_byte) -> std::uint8_t {
    std::uint16_t* o0_p = m.o0_probs.data();
    std::uint16_t* lit_p = m.lit_probs.data() + lit_ctx1(prev_byte, m.lc) * 768u;
    std::uint16_t* o2_p = m.order2_probs.data() +
        static_cast<std::size_t>(
            lit_ctx2(pos, prev_byte, prev2_byte, m.pos_mask, m.o2_mask)) *
            m.o2_stride;
    std::uint16_t* o3_p = m.use_o3
        ? (m.order3_probs.data() +
           lit_ctx3(pos, prev_byte, prev2_byte, prev3_byte, m.pos_mask, m.o3_mask) * 256u)
        : nullptr;
    std::uint16_t* o4_p = m.use_o4
        ? (m.order4_probs.data() +
           lit_ctx4(pos, prev_byte, prev2_byte, prev3_byte, prev4_byte, m.pos_mask, m.o4_mask) * 256u)
        : nullptr;
    const std::int32_t w = m.w_adapt;
    const bool high_noise = (m.lit_gain < 0);
    std::uint32_t raw_byte_cost = 0;
    std::uint32_t sym = 1;
    for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
        if (matched) {
            const int match_bit = (match_byte >> bit_idx) & 1;
            const std::uint32_t idx =
                0x100u + (static_cast<std::uint32_t>(match_bit) << 8) + sym;
            const std::uint32_t p_raw = m.o2_matched
                ? mix3_backoff(o0_p[idx], lit_p[idx], o2_p[idx])
                : mix2_backoff(o0_p[idx], lit_p[idx]);
            const std::uint32_t p_used = scale_prob(p_raw, w);
            const int bit = rd.decode_bit_prob(p_used);
            raw_byte_cost += price_bit(p_raw, bit);
            if (!high_noise) {
                update_prob<5>(o0_p[idx], bit);
                update_prob<4>(lit_p[idx], bit);
                if (m.o2_matched) update_prob<4>(o2_p[idx], bit);
            } else {
                update_prob<7>(o0_p[idx], bit);
                update_prob<7>(lit_p[idx], bit);
            }
            sym = (sym << 1) | static_cast<std::uint32_t>(bit);
            if (bit != match_bit) matched = false;
        } else {
            std::uint32_t p_raw;
            if (o4_p) {
                p_raw = mix5_backoff(o0_p[sym], lit_p[sym], o2_p[sym], o3_p[sym], o4_p[sym]);
            } else if (o3_p) {
                p_raw = mix4_backoff(o0_p[sym], lit_p[sym], o2_p[sym], o3_p[sym]);
            } else {
                p_raw = mix3_backoff(o0_p[sym], lit_p[sym], o2_p[sym]);
            }
            const std::uint32_t p_used = scale_prob(p_raw, w);
            const int bit = rd.decode_bit_prob(p_used);
            raw_byte_cost += price_bit(p_raw, bit);
            if (!high_noise) {
                update_prob<5>(o0_p[sym], bit);
                update_prob<4>(lit_p[sym], bit);
                update_prob<4>(o2_p[sym], bit);
                if (o3_p) update_prob<4>(o3_p[sym], bit);
                if (o4_p) update_prob<4>(o4_p[sym], bit);
            } else {
                update_prob<7>(o0_p[sym], bit);
                update_prob<7>(lit_p[sym], bit);
            }
            sym = (sym << 1) | static_cast<std::uint32_t>(bit);
        }
    }
    update_lit_gain(m, raw_byte_cost);
    return static_cast<std::uint8_t>(sym & 0xFFu);
}

inline auto price_literal(const ModelState& price_m, const ModelState& lit_m,
                          std::uint32_t pos, std::uint8_t prev_byte,
                          std::uint8_t prev2_byte, std::uint8_t prev3_byte,
                          std::uint8_t prev4_byte, bool matched,
                          std::uint8_t match_byte, std::uint8_t symbol_byte)
    -> std::uint32_t {
    const std::uint16_t* o0_p = price_m.o0_probs.data();
    const std::uint16_t* lit_p = lit_m.lit_probs.data() + lit_ctx1(prev_byte, lit_m.lc) * 768u;
    const std::uint16_t* o2_p = lit_m.order2_probs.data() +
        static_cast<std::size_t>(
            lit_ctx2(pos, prev_byte, prev2_byte, lit_m.pos_mask, lit_m.o2_mask)) *
            lit_m.o2_stride;
    const std::uint16_t* o3_p = lit_m.use_o3
        ? (lit_m.order3_probs.data() +
           lit_ctx3(pos, prev_byte, prev2_byte, prev3_byte, lit_m.pos_mask, lit_m.o3_mask) * 256u)
        : nullptr;
    const std::uint16_t* o4_p = lit_m.use_o4
        ? (lit_m.order4_probs.data() +
           lit_ctx4(pos, prev_byte, prev2_byte, prev3_byte, prev4_byte, lit_m.pos_mask, lit_m.o4_mask) * 256u)
        : nullptr;
    const std::int32_t w = price_m.w_adapt;
    std::uint32_t price = 0;
    std::uint32_t sym = 1;
    for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
        const int bit = (symbol_byte >> bit_idx) & 1;
        if (matched) {
            const int match_bit = (match_byte >> bit_idx) & 1;
            const std::uint32_t idx =
                0x100u + (static_cast<std::uint32_t>(match_bit) << 8) + sym;
            const std::uint32_t p_raw = lit_m.o2_matched
                ? mix3_backoff(o0_p[idx], lit_p[idx], o2_p[idx])
                : mix2_backoff(o0_p[idx], lit_p[idx]);
            price += price_bit(scale_prob(p_raw, w), bit);
            sym = (sym << 1) | static_cast<std::uint32_t>(bit);
            if (bit != match_bit) matched = false;
        } else {
            std::uint32_t p_raw;
            if (o4_p) {
                p_raw = mix5_backoff(o0_p[sym], lit_p[sym], o2_p[sym], o3_p[sym], o4_p[sym]);
            } else if (o3_p) {
                p_raw = mix4_backoff(o0_p[sym], lit_p[sym], o2_p[sym], o3_p[sym]);
            } else {
                p_raw = mix3_backoff(o0_p[sym], lit_p[sym], o2_p[sym]);
            }
            price += price_bit(scale_prob(p_raw, w), bit);
            sym = (sym << 1) | static_cast<std::uint32_t>(bit);
        }
    }
    return price;
}

inline void encode_dist(RangeEncoder& rc, ModelState& m, std::uint32_t pos,
                        std::uint32_t len, std::uint32_t dist) {
    const std::uint32_t len_state = len_to_dist_state(len, m.use_len8);
    const std::uint32_t slot = get_pos_slot(dist);
    std::uint32_t tree_m = 1u;
    std::uint32_t prefix = 0u;
    for (int i = 5; i >= 0; --i) {
        const int bit = static_cast<int>((slot >> i) & 1u);
        const std::uint32_t min_slot_if_1 = ((prefix << 1) | 1u) << i;
        if (min_dist_for_slot(min_slot_if_1) <= pos) {
            const std::uint32_t p = mix2_backoff(
                m.slot_o0_coder[tree_m], m.slot_coder[len_state][tree_m]);
            rc.encode_bit_prob(p, bit);
            update_prob<5>(m.slot_o0_coder[tree_m], bit);
            update_prob<4>(m.slot_coder[len_state][tree_m], bit);
        }
        tree_m = (tree_m << 1) | static_cast<std::uint32_t>(bit);
        prefix = (prefix << 1) | static_cast<std::uint32_t>(bit);
    }
    if (slot >= 4u) {
        const int num_extra = static_cast<int>(slot >> 1) - 1;
        const std::uint32_t base = (2u | (slot & 1u)) << num_extra;
        const std::uint32_t extra = (dist - 1u) - base;
        if (slot < 14u) {
            encode_reverse_bit_tree(rc, m.special_coder.data() + (base - slot),
                                    num_extra, extra);
        } else {
            rc.encode_direct_bits(extra >> 4, num_extra - 4);
            encode_reverse_bit_tree(rc, m.align_coder.data(), 4, extra & 0xFu);
        }
    }
}

inline auto decode_dist(RangeDecoder& rd, ModelState& m, std::uint32_t pos,
                        std::uint32_t len) -> std::uint32_t {
    const std::uint32_t len_state = len_to_dist_state(len, m.use_len8);
    std::uint32_t tree_m = 1u;
    std::uint32_t prefix = 0u;
    for (int i = 5; i >= 0; --i) {
        const std::uint32_t min_slot_if_1 = ((prefix << 1) | 1u) << i;
        int bit = 0;
        if (min_dist_for_slot(min_slot_if_1) <= pos) {
            const std::uint32_t p = mix2_backoff(
                m.slot_o0_coder[tree_m], m.slot_coder[len_state][tree_m]);
            bit = rd.decode_bit_prob(p);
            update_prob<5>(m.slot_o0_coder[tree_m], bit);
            update_prob<4>(m.slot_coder[len_state][tree_m], bit);
        }
        tree_m = (tree_m << 1) | static_cast<std::uint32_t>(bit);
        prefix = (prefix << 1) | static_cast<std::uint32_t>(bit);
    }
    const std::uint32_t slot = prefix;
    if (slot < 4u) {
        return slot + 1u;
    }
    const int num_extra = static_cast<int>(slot >> 1) - 1;
    std::uint32_t d = (2u | (slot & 1u)) << num_extra;
    if (slot < 14u) {
        d += decode_reverse_bit_tree(rd, m.special_coder.data() + (d - slot),
                                     num_extra);
    } else {
        d += (rd.decode_direct_bits(num_extra - 4) << 4);
        d += decode_reverse_bit_tree(rd, m.align_coder.data(), 4);
    }
    return d + 1u;
}

inline auto price_dist_direct(const ModelState& m, std::uint32_t pos,
                              std::uint32_t len, std::uint32_t dist)
    -> std::uint32_t {
    const std::uint32_t len_state = len_to_dist_state(len, m.use_len8);
    const std::uint32_t slot = get_pos_slot(dist);
    std::uint32_t price = 0;
    std::uint32_t tree_m = 1u;
    std::uint32_t prefix = 0u;
    for (int i = 5; i >= 0; --i) {
        const int bit = static_cast<int>((slot >> i) & 1u);
        const std::uint32_t min_slot_if_1 = ((prefix << 1) | 1u) << i;
        if (min_dist_for_slot(min_slot_if_1) <= pos) {
            const std::uint32_t p = mix2_backoff(
                m.slot_o0_coder[tree_m], m.slot_coder[len_state][tree_m]);
            price += price_bit(p, bit);
        }
        tree_m = (tree_m << 1) | static_cast<std::uint32_t>(bit);
        prefix = (prefix << 1) | static_cast<std::uint32_t>(bit);
    }
    if (slot >= 4u) {
        const int num_extra = static_cast<int>(slot >> 1) - 1;
        const std::uint32_t base = (2u | (slot & 1u)) << num_extra;
        const std::uint32_t extra = (dist - 1u) - base;
        if (slot < 14u) {
            price += price_reverse_bit_tree(
                m.special_coder.data() + (base - slot), num_extra, extra);
        } else {
            price += static_cast<std::uint32_t>(num_extra - 4) * 64u;
            price += price_reverse_bit_tree(m.align_coder.data(), 4, extra & 0xFu);
        }
    }
    return price;
}

// ---------------------------------------------------------------------------
// Precomputed Price Tables for Fast O(1) Optimal Parser Queries
// ---------------------------------------------------------------------------
struct PriceTables {
    std::array<std::array<std::uint32_t, kNumLenSymbols>, 16> match_len{};
    std::array<std::array<std::uint32_t, kNumLenSymbols>, 16> rep_len{};
    std::array<std::array<std::array<std::uint16_t, 2>, 64>, kNumLenStates> slot_bit_prices{};
    std::array<std::uint32_t, 128> small_dist_extra{};
    std::array<std::uint32_t, 16> align{};
    bool use_len8 = false;

    void update(const ModelState& m) {
        use_len8 = m.use_len8;
        const std::uint32_t num_ps = 1u << m.pb;
        update_len_table(m.match_len_coder, num_ps, kMinMatchLen, match_len);
        update_len_table(m.rep_len_coder, num_ps, kMinRepLen, rep_len);

        const std::uint32_t n_ls = use_len8 ? 8u : 4u;
        for (std::uint32_t ls = 0; ls < n_ls; ++ls) {
            for (std::uint32_t tm = 1; tm < 64u; ++tm) {
                const std::uint32_t p = mix2_backoff(
                    m.slot_o0_coder[tm], m.slot_coder[ls][tm]);
                slot_bit_prices[ls][tm][0] = static_cast<std::uint16_t>(price_bit(p, 0));
                slot_bit_prices[ls][tm][1] = static_cast<std::uint16_t>(price_bit(p, 1));
            }
        }
        for (std::uint32_t d = 0; d < 4; ++d) {
            small_dist_extra[d] = 0;
        }
        for (std::uint32_t d = 4; d < 128; ++d) {
            const std::uint32_t s = get_pos_slot(d + 1u);
            const int num_extra = static_cast<int>(s >> 1) - 1;
            const std::uint32_t base = (2u | (s & 1u)) << num_extra;
            small_dist_extra[d] = price_reverse_bit_tree(
                m.special_coder.data() + (base - s), num_extra, d - base);
        }
        for (std::uint32_t a = 0; a < 16; ++a) {
            align[a] = price_reverse_bit_tree(m.align_coder.data(), 4, a);
        }
    }

    auto price_match_len(std::uint32_t ps, std::uint32_t len) const -> std::uint32_t {
        return match_len[ps][len - kMinMatchLen];
    }

    auto price_rep_len(std::uint32_t ps, std::uint32_t len) const -> std::uint32_t {
        return rep_len[ps][len - kMinRepLen];
    }

    auto price_dist(std::uint32_t len, std::uint32_t dist,
                    std::uint32_t pos) const -> std::uint32_t {
        const std::uint32_t ls = len_to_dist_state(len, use_len8);
        const std::uint32_t d = dist - 1u;
        const std::uint32_t slot = get_pos_slot(dist);
        std::uint32_t price = 0;
        std::uint32_t tree_m = 1u;
        std::uint32_t prefix = 0u;
        for (int i = 5; i >= 0; --i) {
            const int bit = static_cast<int>((slot >> i) & 1u);
            const std::uint32_t min_slot_if_1 = ((prefix << 1) | 1u) << i;
            if (min_dist_for_slot(min_slot_if_1) <= pos) {
                price += slot_bit_prices[ls][tree_m][bit];
            }
            tree_m = (tree_m << 1) | static_cast<std::uint32_t>(bit);
            prefix = (prefix << 1) | static_cast<std::uint32_t>(bit);
        }
        if (d < 4u) return price;
        if (d < 128u) return price + small_dist_extra[d];
        const int num_extra = static_cast<int>(slot >> 1) - 1;
        return price +
               static_cast<std::uint32_t>(num_extra - 4) * 64u +
               align[d & 15u];
    }

  private:
    static void update_len_table(
        const LenCoder& lc, std::uint32_t num_ps, std::uint32_t min_len,
        std::array<std::array<std::uint32_t, kNumLenSymbols>, 16>& out) {
        const std::uint32_t p0 = price_bit(lc.choice, 0);
        const std::uint32_t p1 = price_bit(lc.choice, 1);
        const std::uint32_t p10 = p1 + price_bit(lc.choice2, 0);
        const std::uint32_t p11 = p1 + price_bit(lc.choice2, 1);
        const std::uint32_t num_syms = kMaxMatchLen - min_len + 1u;
        std::array<std::uint32_t, 256> high_p{};
        for (std::uint32_t s = 0; s < 256 && (s + 16u) < num_syms; ++s) {
            high_p[s] = p11 + price_bit_tree<8>(lc.high.data(), s);
        }
        for (std::uint32_t ps = 0; ps < num_ps; ++ps) {
            for (std::uint32_t s = 0; s < 8; ++s) {
                out[ps][s] = p0 + price_bit_tree<3>(lc.low[ps].data(), s);
                out[ps][8 + s] = p10 + price_bit_tree<3>(lc.mid[ps].data(), s);
            }
            for (std::uint32_t s = 16; s < num_syms; ++s) {
                out[ps][s] = high_p[s - 16];
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Reversible x86 BCJ (E8/E9) Filter (100% Bijective 25-bit Signed Domain)
// ---------------------------------------------------------------------------
inline auto is_bcj_candidate(std::uint32_t v) -> bool {
    // True iff top byte is 0x00 or 0xFF (25-bit signed range [-16 MiB, +16 MiB)).
    return ((v + 0x01000000u) & 0xFE000000u) == 0u;
}

inline auto sign_extend_25(std::uint32_t u) -> std::uint32_t {
    u &= 0x01FFFFFFu;
    return (u & 0x01000000u) ? (u | 0xFF000000u) : u;
}

auto should_enable_bcj(const std::uint8_t* data, std::size_t n) -> bool {
    if (n < 64) return false;
    std::array<std::uint32_t, 1024> raw_seen{};
    std::array<std::uint32_t, 1024> abs_seen{};
    raw_seen.fill(kNullPos);
    abs_seen.fill(kNullPos);

    std::uint32_t raw_dups = 0;
    std::uint32_t abs_dups = 0;
    std::size_t last_unconverted = 0;
    bool has_unconverted = false;
    const std::size_t limit = std::min<std::size_t>(n - 5, 2u << 20);
    for (std::size_t i = 0; i <= limit; ++i) {
        if (data[i] != 0xE8u && data[i] != 0xE9u) continue;
        if (has_unconverted && (i - last_unconverted) <= 3u) {
            last_unconverted = i;
            continue;
        }
        const std::uint32_t rel = load_u32(data + i + 1);
        if (!is_bcj_candidate(rel)) {
            last_unconverted = i;
            has_unconverted = true;
            continue;
        }

        const std::uint32_t ip = static_cast<std::uint32_t>(i + 5);
        const std::uint32_t abs_val = sign_extend_25(rel + ip);

        const std::uint32_t hr = (rel * 2654435761u) >> 22;
        if (raw_seen[hr] == rel) {
            ++raw_dups;
        } else {
            raw_seen[hr] = rel;
        }

        const std::uint32_t ha = (abs_val * 2654435761u) >> 22;
        if (abs_seen[ha] == abs_val) {
            ++abs_dups;
        } else {
            abs_seen[ha] = abs_val;
        }
        i += 4;
    }
    return abs_dups >= 16u && abs_dups > raw_dups;
}

void bcj_x86_forward(std::uint8_t* data, std::size_t n) {
    if (n < 5) return;
    std::size_t last_unconverted = 0;
    bool has_unconverted = false;
    for (std::size_t i = 0; i + 5 <= n; ++i) {
        if (data[i] != 0xE8u && data[i] != 0xE9u) continue;
        if (has_unconverted && (i - last_unconverted) <= 3u) {
            last_unconverted = i;
            continue;
        }
        const std::uint32_t rel = load_u32(data + i + 1);
        if (!is_bcj_candidate(rel)) {
            last_unconverted = i;
            has_unconverted = true;
            continue;
        }
        const std::uint32_t ip = static_cast<std::uint32_t>(i + 5);
        const std::uint32_t abs_val = sign_extend_25(rel + ip);
        store_u32(data + i + 1, abs_val);
        i += 4;
    }
}

void bcj_x86_inverse(std::uint8_t* data, std::size_t n) {
    if (n < 5) return;
    std::size_t last_unconverted = 0;
    bool has_unconverted = false;
    for (std::size_t i = 0; i + 5 <= n; ++i) {
        if (data[i] != 0xE8u && data[i] != 0xE9u) continue;
        if (has_unconverted && (i - last_unconverted) <= 3u) {
            last_unconverted = i;
            continue;
        }
        const std::uint32_t abs_val = load_u32(data + i + 1);
        if (!is_bcj_candidate(abs_val)) {
            last_unconverted = i;
            has_unconverted = true;
            continue;
        }
        const std::uint32_t ip = static_cast<std::uint32_t>(i + 5);
        const std::uint32_t rel = sign_extend_25(abs_val - ip);
        store_u32(data + i + 1, rel);
        i += 4;
    }
}

// ---------------------------------------------------------------------------
// Header Encoding & Mode Selection
// ---------------------------------------------------------------------------
auto choose_lit_mode(std::size_t n) -> std::uint8_t {
    if (n <= 4096) return 0;
    if (n <= 16384) return 1;
    if (n <= 65536) return 2;
    return 3;
}

auto choose_pos_bits(const std::uint8_t* data, std::size_t n, bool bcj_on)
    -> std::uint8_t {
    if (bcj_on) return 2;
    if (n < 256) return 0;

    // Detect 4-byte periodic structure vs unaligned text/stream.
    const std::size_t sample = std::min<std::size_t>(n, 8192);
    std::array<std::uint32_t, 4> zero_counts{};
    std::array<std::uint32_t, 4> sum_bytes{};
    bool has_null = false;
    for (std::size_t i = 0; i < sample; ++i) {
        const std::uint8_t b = data[i];
        if (b == 0) {
            ++zero_counts[i & 3u];
            has_null = true;
        }
        sum_bytes[i & 3u] += b;
    }
    if (!has_null) {
        // Pure text / source code benefits from pb = 0 (faster warmup).
        return 0;
    }
    const auto [zmin, zmax] = std::minmax_element(zero_counts.begin(), zero_counts.end());
    const auto [smin, smax] = std::minmax_element(sum_bytes.begin(), sum_bytes.end());
    const std::uint32_t lane_len = static_cast<std::uint32_t>(sample / 4);
    if ((*zmax - *zmin) > lane_len / 16u || (*smax - *smin) > lane_len * 12u) {
        return 2;
    }
    return (n >= 65536) ? 2 : 0;
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
    while (true) {
        if (off >= size || shift > 63) {
            throw std::runtime_error(err_msg);
        }
        const std::uint8_t b = in[off++];
        val |= static_cast<std::uint64_t>(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0u) return val;
        shift += 7;
    }
}

void write_header(std::vector<std::byte>& out, std::uint8_t version,
                  std::uint8_t filter_id, std::uint8_t lit_mode,
                  std::uint8_t pb, std::size_t size) {
    out.push_back(static_cast<std::byte>(kMagic0));
    out.push_back(static_cast<std::byte>(kMagic1));
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (version & 0x03u) |
        ((filter_id & 0x03u) << 2) |
        ((lit_mode & 0x03u) << 4) |
        ((pb & 0x03u) << 6));
    out.push_back(static_cast<std::byte>(flags));
    append_varint(out, size);
}

// ---------------------------------------------------------------------------
// Match Finder: Hash2 + Hash3 + Hash6/Hash8 + Hash4-Chain / BT4 Binary Tree
// ---------------------------------------------------------------------------
struct Match {
    std::uint32_t len = 0;
    std::uint32_t dist = 0;
};

struct Hash4Way {
    std::uint32_t p[4] = {kNullPos, kNullPos, kNullPos, kNullPos};
};

class MatchFinder {
  public:
    MatchFinder(const std::uint8_t* data, std::uint32_t n, int level)
        : data_(data), n_(n), use_bt4_(level >= 4),
          use_long_hash_(level >= 4 && n > 524288u) {
        std::size_t max_dist_win = 512u << 10;
        std::size_t max_bt_win = 512u << 10;
        if (level >= 11) {
            max_dist_win = 16u << 20;
            max_bt_win = 8u << 20;
        } else if (level >= 9) {
            max_dist_win = 16u << 20;
            max_bt_win = 4u << 20;
        } else if (level >= 4) {
            max_dist_win = 2u << 20;
            max_bt_win = 2u << 20;
        }

        if (level <= 1) search_depth_ = 4;
        else if (level == 2) search_depth_ = 12;
        else if (level == 3) search_depth_ = 32;
        else if (level == 4) search_depth_ = 16;
        else if (level == 5) search_depth_ = 32;
        else if (level == 6) search_depth_ = 48;
        else if (level == 7) search_depth_ = 32;
        else if (level == 8) search_depth_ = 48;
        else if (level == 9) search_depth_ = 96;
        else if (level == 10) search_depth_ = 128;
        else if (level == 11) search_depth_ = 160;
        else search_depth_ = 256;

        const std::size_t raw_dist_win = std::min<std::size_t>(
            std::max<std::size_t>(n, 64u), max_dist_win);
        const std::uint32_t dist_win = static_cast<std::uint32_t>(std::bit_ceil(raw_dist_win));
        max_dist_ = dist_win - 1u;

        const std::size_t raw_bt_win = std::min<std::size_t>(
            std::max<std::size_t>(n, 64u), max_bt_win);
        bt_win_size_ = static_cast<std::uint32_t>(std::bit_ceil(raw_bt_win));
        bt_win_mask_ = bt_win_size_ - 1u;
        bt_max_dist_ = bt_win_size_ - 1u;

        std::uint32_t h4_bits = 18;
        if (n <= 32768u) h4_bits = 14;
        else if (n <= 262144u) h4_bits = 16;
        else if (level >= 9 && n > (1u << 20)) h4_bits = 20;
        h4_mask_ = (1u << h4_bits) - 1u;

        hash2_.assign(65536u, kNullPos);
        hash3_a_.assign(262144u, kNullPos);
        hash3_b_.assign(262144u, kNullPos);
        hash4_.assign(std::size_t{1} << h4_bits, kNullPos);
        if (use_bt4_) {
            son_.assign(static_cast<std::size_t>(bt_win_size_) * 2u, kNullPos);
        } else {
            son_.assign(bt_win_size_, kNullPos);
        }

        if (use_long_hash_) {
            const std::uint32_t h68_bits = (n > (1u << 20)) ? 18u : 16u;
            h68_mask_ = (1u << h68_bits) - 1u;
            const std::size_t h68_cnt = std::size_t{1} << h68_bits;
            hash6_.resize(h68_cnt);
            hash8_.resize(h68_cnt);
        }
    }

    // Advance without collecting matches (when skipping inside a chosen match).
    void skip(std::uint32_t pos) {
        if (pos + 2u <= n_) {
            hash2_[load_u16(data_ + pos)] = pos;
        }
        if (pos + 3u <= n_) {
            const std::uint32_t k3 = hash3(data_ + pos);
            hash3_b_[k3] = hash3_a_[k3];
            hash3_a_[k3] = pos;
        }
        if (use_long_hash_) {
            if (pos + 6u <= n_) {
                auto& slot = hash6_[hash6(data_ + pos, h68_mask_)];
                slot.p[3] = slot.p[2];
                slot.p[2] = slot.p[1];
                slot.p[1] = slot.p[0];
                slot.p[0] = pos;
            }
            if (pos + 8u <= n_) {
                auto& slot = hash8_[hash8(data_ + pos, h68_mask_)];
                slot.p[3] = slot.p[2];
                slot.p[2] = slot.p[1];
                slot.p[1] = slot.p[0];
                slot.p[0] = pos;
            }
        }
        if (pos + 4u > n_) return;

        if (use_bt4_) {
            const int skip_depth =
                (n_ > (4u << 20)) ? std::min(search_depth_, 6) : std::min(search_depth_, 16);
            bt4_step(pos, /*out=*/nullptr, skip_depth);
        } else {
            const std::uint32_t h4 = hash4(data_ + pos, h4_mask_);
            son_[pos & bt_win_mask_] = hash4_[h4];
            hash4_[h4] = pos;
        }
    }

    // Find Pareto-optimal matches at `pos` (strictly ascending length & dist).
    void find(std::uint32_t pos, std::vector<Match>& out) {
        out.clear();
        if (pos + 3u > n_) {
            if (pos + 2u <= n_) hash2_[load_u16(data_ + pos)] = pos;
            return;
        }

        const std::uint32_t limit = std::min<std::uint32_t>(kMaxMatchLen, n_ - pos);
        const std::uint32_t min_valid = (pos > max_dist_) ? (pos - max_dist_) : 0u;

        bt_scratch_.clear();

        const std::uint16_t k2 = load_u16(data_ + pos);
        const std::uint32_t p2 = hash2_[k2];
        hash2_[k2] = pos;
        if (p2 != kNullPos && p2 >= min_valid &&
            data_[p2 + 2u] == data_[pos + 2u]) {
            const std::uint32_t d2 = pos - p2;
            if (d2 <= 2048u) {
                const std::uint32_t l2 = match_len_fast(data_ + p2, data_ + pos, 3u, limit);
                bt_scratch_.push_back({l2, d2});
            }
        }

        const std::uint32_t k3 = hash3(data_ + pos);
        const std::uint32_t p3_a = hash3_a_[k3];
        const std::uint32_t p3_b = hash3_b_[k3];
        hash3_b_[k3] = p3_a;
        hash3_a_[k3] = pos;
        auto check_h3 = [&](std::uint32_t p3) {
            if (p3 != kNullPos && p3 >= min_valid &&
                data_[p3] == data_[pos] && data_[p3 + 1] == data_[pos + 1] &&
                data_[p3 + 2] == data_[pos + 2]) {
                const std::uint32_t d3 = pos - p3;
                if (d3 <= 65536u) {
                    const std::uint32_t l3 = match_len_fast(data_ + p3, data_ + pos, 3u, limit);
                    bt_scratch_.push_back({l3, d3});
                }
            }
        };
        check_h3(p3_a);
        check_h3(p3_b);

        if (use_long_hash_) {
            if (pos + 6u <= n_) {
                auto& slot = hash6_[hash6(data_ + pos, h68_mask_)];
                const Hash4Way prev = slot;
                slot.p[3] = prev.p[2];
                slot.p[2] = prev.p[1];
                slot.p[1] = prev.p[0];
                slot.p[0] = pos;
                for (std::uint32_t p6 : prev.p) {
                    if (p6 != kNullPos && p6 >= min_valid &&
                        load_u32(data_ + p6) == load_u32(data_ + pos) &&
                        load_u16(data_ + p6 + 4u) == load_u16(data_ + pos + 4u)) {
                        const std::uint32_t l6 = match_len_fast(data_ + p6, data_ + pos, 6u, limit);
                        bt_scratch_.push_back({l6, pos - p6});
                    }
                }
            }
            if (pos + 8u <= n_) {
                auto& slot = hash8_[hash8(data_ + pos, h68_mask_)];
                const Hash4Way prev = slot;
                slot.p[3] = prev.p[2];
                slot.p[2] = prev.p[1];
                slot.p[1] = prev.p[0];
                slot.p[0] = pos;
                for (std::uint32_t p8 : prev.p) {
                    if (p8 != kNullPos && p8 >= min_valid &&
                        load_u64(data_ + p8) == load_u64(data_ + pos)) {
                        const std::uint32_t l8 = match_len_fast(data_ + p8, data_ + pos, 8u, limit);
                        bt_scratch_.push_back({l8, pos - p8});
                    }
                }
            }
        }

        if (pos + 4u <= n_) {
            const std::uint32_t bt_min_valid =
                (pos > bt_max_dist_) ? (pos - bt_max_dist_) : 0u;
            if (use_bt4_) {
                bt4_step(pos, &bt_scratch_, search_depth_);
            } else {
                hash4_chain(pos, limit, bt_min_valid, bt_scratch_);
            }
        }

        if (bt_scratch_.empty()) return;

        std::sort(bt_scratch_.begin(), bt_scratch_.end(),
                  [](const Match& a, const Match& b) {
                      if (a.dist != b.dist) return a.dist < b.dist;
                      return a.len > b.len;
                  });

        std::uint32_t best_len = 2u;
        for (const auto& c : bt_scratch_) {
            if (c.len > best_len) {
                if (!out.empty() && out.back().dist == c.dist) {
                    out.back().len = c.len;
                } else {
                    out.push_back(c);
                }
                best_len = c.len;
            }
        }
    }

  private:
    void hash4_chain(std::uint32_t pos, std::uint32_t limit,
                     std::uint32_t min_valid, std::vector<Match>& out) {
        const std::uint32_t h4 = hash4(data_ + pos, h4_mask_);
        std::uint32_t cur = hash4_[h4];
        hash4_[h4] = pos;
        son_[pos & bt_win_mask_] = cur;

        std::uint32_t best_len = 3u;
        int depth = search_depth_;
        while (cur != kNullPos && cur >= min_valid && depth-- > 0) {
            if (load_u32(data_ + cur) == load_u32(data_ + pos)) {
                const std::uint32_t l = match_len_fast(data_ + cur, data_ + pos, 4u, limit);
                if (l > best_len) {
                    best_len = l;
                    out.push_back({l, pos - cur});
                    if (l == limit) break;
                }
            }
            cur = son_[cur & bt_win_mask_];
        }
    }

    void bt4_step(std::uint32_t pos, std::vector<Match>* out, int max_depth) {
        const std::uint32_t limit = std::min<std::uint32_t>(kMaxMatchLen, n_ - pos);
        const std::uint32_t bt_min_valid =
            (pos > bt_max_dist_) ? (pos - bt_max_dist_) : 0u;
        const std::uint32_t h4 = hash4(data_ + pos, h4_mask_);
        std::uint32_t cur = hash4_[h4];
        hash4_[h4] = pos;

        const std::uint32_t cyclic = pos & bt_win_mask_;
        std::uint32_t* ptr0 = &son_[static_cast<std::size_t>(cyclic) * 2u + 0u];
        std::uint32_t* ptr1 = &son_[static_cast<std::size_t>(cyclic) * 2u + 1u];
        std::uint32_t len0 = 0;
        std::uint32_t len1 = 0;
        std::uint32_t best_len = 3u;
        int depth = max_depth;

        for (;;) {
            if (cur == kNullPos || cur < bt_min_valid || depth-- <= 0) {
                *ptr0 = kNullPos;
                *ptr1 = kNullPos;
                break;
            }

            const std::uint32_t m_cyclic = cur & bt_win_mask_;
            std::uint32_t len = std::min(len0, len1);
            len = match_len_fast(data_ + cur, data_ + pos, len, limit);

            if (out != nullptr && len > best_len) {
                best_len = len;
                out->push_back({len, pos - cur});
            }

            if (len == limit) {
                *ptr0 = son_[static_cast<std::size_t>(m_cyclic) * 2u + 0u];
                *ptr1 = son_[static_cast<std::size_t>(m_cyclic) * 2u + 1u];
                break;
            }

            if (data_[cur + len] < data_[pos + len]) {
                *ptr0 = cur;
                ptr0 = &son_[static_cast<std::size_t>(m_cyclic) * 2u + 1u];
                cur = *ptr0;
                len0 = len;
            } else {
                *ptr1 = cur;
                ptr1 = &son_[static_cast<std::size_t>(m_cyclic) * 2u + 0u];
                cur = *ptr1;
                len1 = len;
            }
        }
    }

    const std::uint8_t* data_;
    std::uint32_t n_;
    bool use_bt4_;
    bool use_long_hash_;
    int search_depth_ = 32;
    std::uint32_t bt_win_size_ = 0;
    std::uint32_t bt_win_mask_ = 0;
    std::uint32_t bt_max_dist_ = 0;
    std::uint32_t max_dist_ = 0;
    std::uint32_t h4_mask_ = 0;
    std::uint32_t h68_mask_ = 0;

    std::vector<std::uint32_t> hash2_;
    std::vector<std::uint32_t> hash3_a_;
    std::vector<std::uint32_t> hash3_b_;
    std::vector<std::uint32_t> hash4_;
    std::vector<Hash4Way> hash6_;
    std::vector<Hash4Way> hash8_;
    std::vector<std::uint32_t> son_;
    std::vector<Match> bt_scratch_;
};

// ---------------------------------------------------------------------------
// Token & Direct Token Emission / Costing
// ---------------------------------------------------------------------------
enum TokenType : std::uint8_t {
    kTokLit = 0,
    kTokShortRep = 1,
    kTokRep = 2,
    kTokMatch = 3,
};

struct Token {
    TokenType type = kTokLit;
    std::uint32_t len = 1;
    std::uint32_t dist_or_rep = 0;  // dist for Match, rep_idx for Rep
};

struct ParserState {
    std::uint8_t state = 0;
    std::array<std::uint32_t, 4> rep = {1, 1, 1, 1};
};

inline void emit_token(RangeEncoder& rc, ModelState& m, ParserState& ps,
                       const std::uint8_t* data, std::uint32_t pos,
                       const Token& tok) {
    const std::uint32_t pos_state = pos & m.pos_mask;
    const std::uint8_t prev_b = (pos > 0) ? data[pos - 1] : 0;
    const std::uint8_t prev2_b = (pos > 1) ? data[pos - 2] : 0;
    const std::uint8_t prev3_b = (pos > 2) ? data[pos - 3] : 0;
    const std::uint8_t prev4_b = (pos > 3) ? data[pos - 4] : 0;
    const std::uint32_t match_ctx = match_ctx_of(pos, m.pos_mask, prev_b);
    const std::uint32_t rep_ctx = rep_ctx_of(pos, m.pos_mask, prev_b);

    switch (tok.type) {
        case kTokLit: {
            if (pos > 0) {
                rc.encode_dual(m.is_match[ps.state][match_ctx], 0);
            }
            const bool matched = (ps.state >= 7);
            const std::uint8_t mb = matched ? data[pos - ps.rep[0]] : 0;
            encode_literal(rc, m, pos, prev_b, prev2_b, prev3_b, prev4_b,
                           matched, mb, data[pos]);
            ps.state = kNextStateLit[ps.state];
            break;
        }
        case kTokShortRep: {
            rc.encode_dual(m.is_match[ps.state][match_ctx], 1);
            rc.encode_dual(m.is_rep[ps.state], 1);
            rc.encode_dual(m.is_rep0[ps.state], 0);
            rc.encode_dual(m.is_rep0_long[ps.state][rep_ctx], 0);
            ps.state = state_after_shortrep(ps.state);
            break;
        }
        case kTokRep: {
            const std::uint32_t r = tok.dist_or_rep;
            rc.encode_dual(m.is_match[ps.state][match_ctx], 1);
            rc.encode_dual(m.is_rep[ps.state], 1);
            if (r == 0u) {
                rc.encode_dual(m.is_rep0[ps.state], 0);
                rc.encode_dual(m.is_rep0_long[ps.state][rep_ctx], 1);
            } else {
                rc.encode_dual(m.is_rep0[ps.state], 1);
                if (r == 1u) {
                    rc.encode_dual(m.is_rep1[ps.state], 0);
                    std::swap(ps.rep[0], ps.rep[1]);
                } else if (r == 2u) {
                    rc.encode_dual(m.is_rep1[ps.state], 1);
                    rc.encode_dual(m.is_rep2[ps.state], 0);
                    const std::uint32_t d = ps.rep[2];
                    ps.rep[2] = ps.rep[1];
                    ps.rep[1] = ps.rep[0];
                    ps.rep[0] = d;
                } else {
                    rc.encode_dual(m.is_rep1[ps.state], 1);
                    rc.encode_dual(m.is_rep2[ps.state], 1);
                    const std::uint32_t d = ps.rep[3];
                    ps.rep[3] = ps.rep[2];
                    ps.rep[2] = ps.rep[1];
                    ps.rep[1] = ps.rep[0];
                    ps.rep[0] = d;
                }
            }
            m.rep_len_coder.encode(rc, pos_state, tok.len, kMinRepLen);
            ps.state = state_after_rep(ps.state);
            break;
        }
        case kTokMatch: {
            const std::uint32_t dist = tok.dist_or_rep;
            rc.encode_dual(m.is_match[ps.state][match_ctx], 1);
            rc.encode_dual(m.is_rep[ps.state], 0);
            m.match_len_coder.encode(rc, pos_state, tok.len, kMinMatchLen);
            encode_dist(rc, m, pos, tok.len, dist);
            ps.rep[3] = ps.rep[2];
            ps.rep[2] = ps.rep[1];
            ps.rep[1] = ps.rep[0];
            ps.rep[0] = dist;
            ps.state = state_after_match(ps.state);
            break;
        }
    }
}

// Dry-run bit cost + exact adaptive model state simulation for Pass 2 comparison.
class BitCostSimulator {
  public:
    void encode_bit(std::uint16_t& prob, int bit) {
        total_price_ += price_bit(prob, bit);
        if (bit == 0) {
            prob = static_cast<std::uint16_t>(prob + ((kBitModelTotal - prob) >> 5));
        } else {
            prob = static_cast<std::uint16_t>(prob - (prob >> 5));
        }
    }

    template <int ShiftA, int ShiftB>
    void encode_bit_mixed(std::uint16_t& pa, std::uint16_t& pb, int bit) {
        total_price_ += price_bit_mixed(pa, pb, bit);
        if (bit == 0) {
            pa = static_cast<std::uint16_t>(pa + ((kBitModelTotal - pa) >> ShiftA));
            pb = static_cast<std::uint16_t>(pb + ((kBitModelTotal - pb) >> ShiftB));
        } else {
            pa = static_cast<std::uint16_t>(pa - (pa >> ShiftA));
            pb = static_cast<std::uint16_t>(pb - (pb >> ShiftB));
        }
    }

    void encode_bit_prob(std::uint32_t p, int bit) {
        total_price_ += price_bit(p, bit);
    }

    void encode_dual(DualProb& dp, int bit) {
        encode_bit_mixed<4, 6>(dp.fast, dp.slow, bit);
    }

    void encode_direct_bits(std::uint32_t /*value*/, int num_bits) {
        total_price_ += static_cast<std::uint64_t>(num_bits) * 64u;
    }

    auto total_price() const -> std::uint64_t { return total_price_; }

  private:
    std::uint64_t total_price_ = 0;
};

template <typename Coder>
inline void sim_bit_tree(Coder& c, std::uint16_t* probs, int num_bits,
                         std::uint32_t sym) {
    std::uint32_t m = 1;
    for (int i = num_bits - 1; i >= 0; --i) {
        const int bit = static_cast<int>((sym >> i) & 1u);
        c.encode_bit(probs[m], bit);
        m = (m << 1) | static_cast<std::uint32_t>(bit);
    }
}

inline void sim_reverse_bit_tree(BitCostSimulator& c, std::uint16_t* probs,
                                 int num_bits, std::uint32_t sym) {
    std::uint32_t m = 1;
    for (int i = 0; i < num_bits; ++i) {
        const int bit = static_cast<int>((sym >> i) & 1u);
        c.encode_bit(probs[m], bit);
        m = (m << 1) | static_cast<std::uint32_t>(bit);
    }
}

inline void sim_len(BitCostSimulator& c, LenCoder& lc, std::uint32_t pos_state,
                    std::uint32_t len, std::uint32_t min_len) {
    const std::uint32_t sym = len - min_len;
    if (sym < 8u) {
        c.encode_bit(lc.choice, 0);
        sim_bit_tree(c, lc.low[pos_state].data(), 3, sym);
    } else if (sym < 16u) {
        c.encode_bit(lc.choice, 1);
        c.encode_bit(lc.choice2, 0);
        sim_bit_tree(c, lc.mid[pos_state].data(), 3, sym - 8u);
    } else {
        c.encode_bit(lc.choice, 1);
        c.encode_bit(lc.choice2, 1);
        sim_bit_tree(c, lc.high.data(), 8, sym - 16u);
    }
}

inline void sim_dist(BitCostSimulator& c, ModelState& m, std::uint32_t pos,
                     std::uint32_t len, std::uint32_t dist) {
    const std::uint32_t len_state = len_to_dist_state(len, m.use_len8);
    const std::uint32_t slot = get_pos_slot(dist);
    std::uint32_t tree_m = 1u;
    std::uint32_t prefix = 0u;
    for (int i = 5; i >= 0; --i) {
        const int bit = static_cast<int>((slot >> i) & 1u);
        const std::uint32_t min_slot_if_1 = ((prefix << 1) | 1u) << i;
        if (min_dist_for_slot(min_slot_if_1) <= pos) {
            const std::uint32_t p = mix2_backoff(
                m.slot_o0_coder[tree_m], m.slot_coder[len_state][tree_m]);
            c.encode_bit_prob(p, bit);
            update_prob<5>(m.slot_o0_coder[tree_m], bit);
            update_prob<4>(m.slot_coder[len_state][tree_m], bit);
        }
        tree_m = (tree_m << 1) | static_cast<std::uint32_t>(bit);
        prefix = (prefix << 1) | static_cast<std::uint32_t>(bit);
    }
    if (slot >= 4u) {
        const int num_extra = static_cast<int>(slot >> 1) - 1;
        const std::uint32_t base = (2u | (slot & 1u)) << num_extra;
        const std::uint32_t extra = (dist - 1u) - base;
        if (slot < 14u) {
            sim_reverse_bit_tree(c, m.special_coder.data() + (base - slot),
                                 num_extra, extra);
        } else {
            c.encode_direct_bits(extra >> 4, num_extra - 4);
            sim_reverse_bit_tree(c, m.align_coder.data(), 4, extra & 0xFu);
        }
    }
}

struct ProbUndo {
    std::uint16_t* ptr;
    std::uint16_t old_val;
};

auto simulate_chunk(ModelState& core_m, ModelState& lit_m, ParserState ps,
                    const std::uint8_t* data, std::uint32_t start_pos,
                    const std::vector<Token>& tokens,
                    std::vector<ProbUndo>& undo_log) -> std::uint64_t {
    BitCostSimulator sim;
    undo_log.clear();

    auto sim_update_lit4 = [&](std::uint16_t& p, int bit) {
        undo_log.push_back({&p, p});
        update_prob<4>(p, bit);
    };
    auto sim_update_lit7 = [&](std::uint16_t& p, int bit) {
        undo_log.push_back({&p, p});
        update_prob<7>(p, bit);
    };

    std::uint32_t pos = start_pos;
    for (const auto& tok : tokens) {
        const std::uint32_t pos_state = pos & core_m.pos_mask;
        const std::uint8_t prev_b = (pos > 0) ? data[pos - 1] : 0;
        const std::uint8_t prev2_b = (pos > 1) ? data[pos - 2] : 0;
        const std::uint8_t prev3_b = (pos > 2) ? data[pos - 3] : 0;
        const std::uint8_t prev4_b = (pos > 3) ? data[pos - 4] : 0;
        const std::uint32_t match_ctx = match_ctx_of(pos, core_m.pos_mask, prev_b);
        const std::uint32_t rep_ctx = rep_ctx_of(pos, core_m.pos_mask, prev_b);

        switch (tok.type) {
            case kTokLit: {
                if (pos > 0) {
                    sim.encode_dual(core_m.is_match[ps.state][match_ctx], 0);
                }
                bool matched = (ps.state >= 7);
                const std::uint8_t mb = matched ? data[pos - ps.rep[0]] : 0;
                std::uint16_t* o0_p = core_m.o0_probs.data();
                std::uint16_t* lit_p = lit_m.lit_probs.data() + lit_ctx1(prev_b, lit_m.lc) * 768u;
                std::uint16_t* o2_p = lit_m.order2_probs.data() +
                    static_cast<std::size_t>(
                        lit_ctx2(pos, prev_b, prev2_b, lit_m.pos_mask, lit_m.o2_mask)) *
                        lit_m.o2_stride;
                std::uint16_t* o3_p = lit_m.use_o3
                    ? (lit_m.order3_probs.data() +
                       lit_ctx3(pos, prev_b, prev2_b, prev3_b, lit_m.pos_mask, lit_m.o3_mask) * 256u)
                    : nullptr;
                std::uint16_t* o4_p = lit_m.use_o4
                    ? (lit_m.order4_probs.data() +
                       lit_ctx4(pos, prev_b, prev2_b, prev3_b, prev4_b, lit_m.pos_mask, lit_m.o4_mask) * 256u)
                    : nullptr;
                const std::int32_t w = core_m.w_adapt;
                const bool high_noise = (core_m.lit_gain < 0);
                std::uint32_t raw_byte_cost = 0;
                std::uint32_t sym = 1;
                const std::uint8_t sb = data[pos];
                for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
                    const int bit = (sb >> bit_idx) & 1;
                    if (matched) {
                        const int mbit = (mb >> bit_idx) & 1;
                        const std::uint32_t idx =
                            0x100u + (static_cast<std::uint32_t>(mbit) << 8) + sym;
                        const std::uint32_t p_raw = lit_m.o2_matched
                            ? mix3_backoff(o0_p[idx], lit_p[idx], o2_p[idx])
                            : mix2_backoff(o0_p[idx], lit_p[idx]);
                        const std::uint32_t p_used = scale_prob(p_raw, w);
                        sim.encode_bit_prob(p_used, bit);
                        raw_byte_cost += price_bit(p_raw, bit);
                        if (!high_noise) {
                            update_prob<5>(o0_p[idx], bit);
                            sim_update_lit4(lit_p[idx], bit);
                            if (lit_m.o2_matched) sim_update_lit4(o2_p[idx], bit);
                        } else {
                            update_prob<7>(o0_p[idx], bit);
                            sim_update_lit7(lit_p[idx], bit);
                        }
                        sym = (sym << 1) | static_cast<std::uint32_t>(bit);
                        if (bit != mbit) matched = false;
                    } else {
                        std::uint32_t p_raw;
                        if (o4_p) {
                            p_raw = mix5_backoff(o0_p[sym], lit_p[sym], o2_p[sym], o3_p[sym], o4_p[sym]);
                        } else if (o3_p) {
                            p_raw = mix4_backoff(o0_p[sym], lit_p[sym], o2_p[sym], o3_p[sym]);
                        } else {
                            p_raw = mix3_backoff(o0_p[sym], lit_p[sym], o2_p[sym]);
                        }
                        const std::uint32_t p_used = scale_prob(p_raw, w);
                        sim.encode_bit_prob(p_used, bit);
                        raw_byte_cost += price_bit(p_raw, bit);
                        if (!high_noise) {
                            update_prob<5>(o0_p[sym], bit);
                            sim_update_lit4(lit_p[sym], bit);
                            sim_update_lit4(o2_p[sym], bit);
                            if (o3_p) sim_update_lit4(o3_p[sym], bit);
                            if (o4_p) sim_update_lit4(o4_p[sym], bit);
                        } else {
                            update_prob<7>(o0_p[sym], bit);
                            sim_update_lit7(lit_p[sym], bit);
                        }
                        sym = (sym << 1) | static_cast<std::uint32_t>(bit);
                    }
                }
                update_lit_gain(core_m, raw_byte_cost);
                ps.state = kNextStateLit[ps.state];
                pos += 1u;
                break;
            }
            case kTokShortRep: {
                sim.encode_dual(core_m.is_match[ps.state][match_ctx], 1);
                sim.encode_dual(core_m.is_rep[ps.state], 1);
                sim.encode_dual(core_m.is_rep0[ps.state], 0);
                sim.encode_dual(core_m.is_rep0_long[ps.state][rep_ctx], 0);
                ps.state = state_after_shortrep(ps.state);
                pos += 1u;
                break;
            }
            case kTokRep: {
                const std::uint32_t r = tok.dist_or_rep;
                sim.encode_dual(core_m.is_match[ps.state][match_ctx], 1);
                sim.encode_dual(core_m.is_rep[ps.state], 1);
                if (r == 0u) {
                    sim.encode_dual(core_m.is_rep0[ps.state], 0);
                    sim.encode_dual(core_m.is_rep0_long[ps.state][rep_ctx], 1);
                } else {
                    sim.encode_dual(core_m.is_rep0[ps.state], 1);
                    if (r == 1u) {
                        sim.encode_dual(core_m.is_rep1[ps.state], 0);
                        std::swap(ps.rep[0], ps.rep[1]);
                    } else if (r == 2u) {
                        sim.encode_dual(core_m.is_rep1[ps.state], 1);
                        sim.encode_dual(core_m.is_rep2[ps.state], 0);
                        const std::uint32_t d = ps.rep[2];
                        ps.rep[2] = ps.rep[1];
                        ps.rep[1] = ps.rep[0];
                        ps.rep[0] = d;
                    } else {
                        sim.encode_dual(core_m.is_rep1[ps.state], 1);
                        sim.encode_dual(core_m.is_rep2[ps.state], 1);
                        const std::uint32_t d = ps.rep[3];
                        ps.rep[3] = ps.rep[2];
                        ps.rep[2] = ps.rep[1];
                        ps.rep[1] = ps.rep[0];
                        ps.rep[0] = d;
                    }
                }
                sim_len(sim, core_m.rep_len_coder, pos_state, tok.len, kMinRepLen);
                ps.state = state_after_rep(ps.state);
                pos += tok.len;
                break;
            }
            case kTokMatch: {
                const std::uint32_t dist = tok.dist_or_rep;
                sim.encode_dual(core_m.is_match[ps.state][match_ctx], 1);
                sim.encode_dual(core_m.is_rep[ps.state], 0);
                sim_len(sim, core_m.match_len_coder, pos_state, tok.len, kMinMatchLen);
                sim_dist(sim, core_m, pos, tok.len, dist);
                ps.rep[3] = ps.rep[2];
                ps.rep[2] = ps.rep[1];
                ps.rep[1] = ps.rep[0];
                ps.rep[0] = dist;
                ps.state = state_after_match(ps.state);
                pos += tok.len;
                break;
            }
        }
    }

    for (auto it = undo_log.rbegin(); it != undo_log.rend(); ++it) {
        *it->ptr = it->old_val;
    }
    undo_log.clear();
    return sim.total_price();
}

// ---------------------------------------------------------------------------
// Fast / Lazy Parser (Levels 1..6)
// ---------------------------------------------------------------------------
struct BestChoice {
    Token tok{};
    std::uint32_t cost = 0;
};

auto eval_best_at(const ModelState& m, const ParserState& ps,
                  const std::uint8_t* data, std::uint32_t pos, std::uint32_t n,
                  const std::vector<Match>& matches, bool exact_pricing)
    -> BestChoice {
    const std::uint32_t pos_state = pos & m.pos_mask;
    const std::uint32_t limit = std::min<std::uint32_t>(kMaxMatchLen, n - pos);
    const std::uint8_t prev_b = data[pos - 1];
    const std::uint8_t prev2_b = (pos > 1) ? data[pos - 2] : 0;
    const std::uint8_t prev3_b = (pos > 2) ? data[pos - 3] : 0;
    const std::uint8_t prev4_b = (pos > 3) ? data[pos - 4] : 0;
    const std::uint32_t match_ctx = match_ctx_of(pos, m.pos_mask, prev_b);
    const std::uint32_t rep_ctx = rep_ctx_of(pos, m.pos_mask, prev_b);

    // 1. Literal baseline
    const std::uint32_t p_is_match0 = m.is_match[ps.state][match_ctx].price(0);
    const bool matched = (ps.state >= 7);
    const std::uint8_t mb = matched ? data[pos - ps.rep[0]] : 0;
    const std::uint32_t lit_cost = p_is_match0 +
        price_literal(m, m, pos, prev_b, prev2_b, prev3_b, prev4_b,
                      matched, mb, data[pos]);

    BestChoice best{};
    best.tok = {kTokLit, 1u, 0u};
    best.cost = lit_cost;

    const std::uint32_t p_is_match1 = m.is_match[ps.state][match_ctx].price(1);
    const std::uint32_t p_is_rep0 = m.is_rep[ps.state].price(0);
    const std::uint32_t p_is_rep1 = m.is_rep[ps.state].price(1);

    auto lit_baseline_for = [&](std::uint32_t len) -> std::uint32_t {
        std::uint32_t sum = 0;
        for (std::uint32_t k = 0; k < len; ++k) {
            const std::uint32_t p = pos + k;
            const std::uint8_t pb = data[p - 1];
            const std::uint8_t p2b = (p > 1) ? data[p - 2] : 0;
            const std::uint8_t p3b = (p > 2) ? data[p - 3] : 0;
            const std::uint8_t p4b = (p > 3) ? data[p - 4] : 0;
            sum += m.is_match[ps.state][match_ctx_of(p, m.pos_mask, pb)].price(0) +
                   price_literal(m, m, p, pb, p2b, p3b, p4b, false, 0, data[p]);
        }
        return sum;
    };

    // 2. Check repeat offsets (ShortRep + Rep0..Rep3)
    double best_gain = 0.0;
    if (pos >= ps.rep[0] && data[pos] == data[pos - ps.rep[0]]) {
        const std::uint32_t sr_cost = p_is_match1 + p_is_rep1 +
            m.is_rep0[ps.state].price(0) +
            m.is_rep0_long[ps.state][rep_ctx].price(0);
        if (sr_cost < best.cost) {
            best_gain = static_cast<double>(lit_cost) - static_cast<double>(sr_cost);
            best.tok = {kTokShortRep, 1u, 0u};
            best.cost = sr_cost;
        }
    }

    for (std::uint32_t r = 0; r < 4u; ++r) {
        const std::uint32_t d = ps.rep[r];
        if (d > pos || limit < 2u) continue;
        if (data[pos] != data[pos - d] || data[pos + 1] != data[pos + 1 - d]) {
            continue;
        }
        const std::uint32_t rlen = match_len_fast(data + pos - d, data + pos, 2u, limit);
        std::uint32_t rep_hdr = p_is_match1 + p_is_rep1;
        if (r == 0u) {
            rep_hdr += m.is_rep0[ps.state].price(0) +
                       m.is_rep0_long[ps.state][rep_ctx].price(1);
        } else {
            rep_hdr += m.is_rep0[ps.state].price(1);
            if (r == 1u) {
                rep_hdr += m.is_rep1[ps.state].price(0);
            } else {
                rep_hdr += m.is_rep1[ps.state].price(1) +
                           m.is_rep2[ps.state].price(r == 3u ? 1 : 0);
            }
        }
        const std::uint32_t r_cost = rep_hdr +
            m.rep_len_coder.price_direct(pos_state, rlen, kMinRepLen);
        const std::uint32_t raw_lits = exact_pricing ? lit_baseline_for(rlen) : (rlen * 512u);
        const double gain = static_cast<double>(raw_lits) - static_cast<double>(r_cost);
        if (gain > best_gain) {
            best_gain = gain;
            best.tok = {kTokRep, rlen, r};
            best.cost = r_cost;
        }
    }

    // 3. Check normal matches (len >= 3)
    for (const auto& mt : matches) {
        if (mt.len < 3u || mt.dist > pos) continue;
        const std::uint32_t m_cost = p_is_match1 + p_is_rep0 +
            m.match_len_coder.price_direct(pos_state, mt.len, kMinMatchLen) +
            price_dist_direct(m, pos, mt.len, mt.dist);
        const std::uint32_t raw_lits = exact_pricing ? lit_baseline_for(mt.len) : (mt.len * 512u);
        const double gain = static_cast<double>(raw_lits) - static_cast<double>(m_cost);
        if (gain > best_gain) {
            best_gain = gain;
            best.tok = {kTokMatch, mt.len, mt.dist};
            best.cost = m_cost;
        }
    }

    return best;
}

void compress_fast_lazy(RangeEncoder& rc, ModelState& m, MatchFinder& mf,
                        const std::uint8_t* data, std::uint32_t n, int level) {
    ParserState ps{};
    if (n == 0u) return;

    // pos == 0 is always a literal
    emit_token(rc, m, ps, data, 0u, {kTokLit, 1u, 0u});
    mf.skip(0u);
    if (n == 1u) return;

    const bool use_lazy1 = (level >= 2);
    const bool use_lazy2 = (level >= 3);
    const bool exact_pricing = (level >= 4);

    std::vector<Match> matches;
    std::vector<Match> next_matches;
    std::vector<Match> next2_matches;
    std::uint32_t pos = 1u;
    std::uint32_t mf_cursor = 1u;

    auto find_at = [&](std::uint32_t p, std::vector<Match>& out) {
        while (mf_cursor < p) {
            mf.skip(mf_cursor++);
        }
        if (mf_cursor == p) {
            mf.find(p, out);
            ++mf_cursor;
        } else {
            out.clear();
        }
    };

    while (pos < n) {
        find_at(pos, matches);
        BestChoice cur = eval_best_at(m, ps, data, pos, n, matches, exact_pricing);

        if (cur.tok.type != kTokLit && cur.tok.len >= 2u &&
            use_lazy1 && pos + 1u < n) {
            // Check lazy1 at pos + 1 (advancing as literal at pos)
            ParserState ps1 = ps;
            const std::uint32_t lit_p = m.is_match[ps.state][match_ctx_of(pos, m.pos_mask, data[pos - 1])].price(0) +
                price_literal(m, m, pos, data[pos - 1],
                              (pos > 1u) ? data[pos - 2u] : 0u,
                              (pos > 2u) ? data[pos - 3u] : 0u,
                              (pos > 3u) ? data[pos - 4u] : 0u,
                              ps.state >= 7, (ps.state >= 7) ? data[pos - ps.rep[0]] : 0,
                              data[pos]);
            ps1.state = kNextStateLit[ps.state];

            find_at(pos + 1u, next_matches);
            BestChoice nxt = eval_best_at(m, ps1, data, pos + 1u, n, next_matches, exact_pricing);
            if (nxt.tok.type != kTokLit && nxt.tok.len >= 2u) {
                const bool longer_enough = (nxt.tok.len >= cur.tok.len + 1u) ||
                                           (nxt.tok.len == cur.tok.len &&
                                            lit_p + nxt.cost < cur.cost + 128u);
                if (longer_enough) {
                    emit_token(rc, m, ps, data, pos, {kTokLit, 1u, 0u});
                    ++pos;
                    cur = nxt;
                    // Also check lazy2 at pos + 1
                    if (use_lazy2 && pos + 1u < n) {
                        ParserState ps2 = ps;
                        ps2.state = kNextStateLit[ps.state];
                        find_at(pos + 1u, next2_matches);
                        BestChoice nxt2 = eval_best_at(m, ps2, data, pos + 1u, n, next2_matches, exact_pricing);
                        if (nxt2.tok.type != kTokLit && nxt2.tok.len >= cur.tok.len + 2u) {
                            emit_token(rc, m, ps, data, pos, {kTokLit, 1u, 0u});
                            ++pos;
                            cur = nxt2;
                        }
                    }
                    emit_token(rc, m, ps, data, pos, cur.tok);
                    pos += cur.tok.len;
                    while (mf_cursor < pos) {
                        mf.skip(mf_cursor++);
                    }
                    continue;
                }
            }
        }

        emit_token(rc, m, ps, data, pos, cur.tok);
        pos += cur.tok.len;
        while (mf_cursor < pos) {
            mf.skip(mf_cursor++);
        }
    }
}

// ---------------------------------------------------------------------------
// State-Tracking Forward DP Optimal Parser (Levels 7..12)
// ---------------------------------------------------------------------------
enum OptKind : std::uint8_t {
    kOptLit = 0,
    kOptShortRep = 1,
    kOptRep = 2,
    kOptMatch = 3,
    kOptLitRep0 = 4,       // Lit(back) + Rep0/ShortRep(len2) at back+1
    kOptMatchLitRep0 = 5,  // Match(len, dist) + Lit + Rep0/ShortRep(len2)
    kOptRepLitRep0 = 6,    // Rep(len, r) + Lit + Rep0/ShortRep(len2)
};

struct OptNode {
    std::uint64_t cost = kInfinity;
    std::uint32_t back = 0;
    std::uint32_t dist_or_rep = 0;
    std::uint16_t len = 0;
    std::uint16_t len2 = 0;
    std::uint8_t kind = kOptLit;
    std::uint8_t state = 0;
    std::array<std::uint32_t, 4> rep{};
};

struct MatchSlice {
    std::uint32_t offset = 0;
    std::uint32_t count = 0;
};

void copy_core_model(ModelState& dst, const ModelState& src) {
    dst.is_match = src.is_match;
    dst.is_rep = src.is_rep;
    dst.is_rep0 = src.is_rep0;
    dst.is_rep0_long = src.is_rep0_long;
    dst.is_rep1 = src.is_rep1;
    dst.is_rep2 = src.is_rep2;
    dst.match_len_coder = src.match_len_coder;
    dst.rep_len_coder = src.rep_len_coder;
    dst.slot_o0_coder = src.slot_o0_coder;
    dst.slot_coder = src.slot_coder;
    dst.special_coder = src.special_coder;
    dst.align_coder = src.align_coder;
    dst.lit_gain = src.lit_gain;
    dst.w_adapt = src.w_adapt;
    dst.o0_probs = src.o0_probs;
}

void blend_core_models(ModelState& dst, const ModelState& a, const ModelState& b,
                       int weight_b = 1) {
    auto avg16 = [weight_b](std::uint16_t x, std::uint16_t y) -> std::uint16_t {
        if (weight_b == 3) {
            return static_cast<std::uint16_t>((static_cast<std::uint32_t>(x) +
                                               static_cast<std::uint32_t>(y) * 3u) >> 2);
        }
        return static_cast<std::uint16_t>((static_cast<std::uint32_t>(x) +
                                           static_cast<std::uint32_t>(y)) >> 1);
    };
    auto avg_dual = [&](DualProb& d, const DualProb& x, const DualProb& y) {
        d.fast = avg16(x.fast, y.fast);
        d.slow = avg16(x.slow, y.slow);
    };

    for (std::size_t s = 0; s < kNumStates; ++s) {
        for (std::size_t ps = 0; ps < kNumPosStatesMax; ++ps) {
            avg_dual(dst.is_match[s][ps], a.is_match[s][ps], b.is_match[s][ps]);
            avg_dual(dst.is_rep0_long[s][ps], a.is_rep0_long[s][ps], b.is_rep0_long[s][ps]);
        }
        avg_dual(dst.is_rep[s], a.is_rep[s], b.is_rep[s]);
        avg_dual(dst.is_rep0[s], a.is_rep0[s], b.is_rep0[s]);
        avg_dual(dst.is_rep1[s], a.is_rep1[s], b.is_rep1[s]);
        avg_dual(dst.is_rep2[s], a.is_rep2[s], b.is_rep2[s]);
    }

    auto avg_len = [&](LenCoder& dc, const LenCoder& x, const LenCoder& y) {
        dc.choice = avg16(x.choice, y.choice);
        dc.choice2 = avg16(x.choice2, y.choice2);
        for (std::size_t ps = 0; ps < kNumPosStatesMax; ++ps) {
            for (std::size_t i = 0; i < 8; ++i) {
                dc.low[ps][i] = avg16(x.low[ps][i], y.low[ps][i]);
                dc.mid[ps][i] = avg16(x.mid[ps][i], y.mid[ps][i]);
            }
        }
        for (std::size_t i = 0; i < 256; ++i) {
            dc.high[i] = avg16(x.high[i], y.high[i]);
        }
    };
    avg_len(dst.match_len_coder, a.match_len_coder, b.match_len_coder);
    avg_len(dst.rep_len_coder, a.rep_len_coder, b.rep_len_coder);

    for (std::size_t i = 0; i < 64; ++i) {
        dst.slot_o0_coder[i] = avg16(a.slot_o0_coder[i], b.slot_o0_coder[i]);
    }
    for (std::size_t ls = 0; ls < kNumLenStates; ++ls) {
        for (std::size_t i = 0; i < 64; ++i) {
            dst.slot_coder[ls][i] = avg16(a.slot_coder[ls][i], b.slot_coder[ls][i]);
        }
    }
    for (std::size_t i = 0; i < 128; ++i) {
        dst.special_coder[i] = avg16(a.special_coder[i], b.special_coder[i]);
    }
    for (std::size_t i = 0; i < 16; ++i) {
        dst.align_coder[i] = avg16(a.align_coder[i], b.align_coder[i]);
    }
    if (weight_b == 3) {
        dst.lit_gain = (a.lit_gain + b.lit_gain * 3) / 4;
        dst.w_adapt = (a.w_adapt + b.w_adapt * 3) / 4;
    } else {
        dst.lit_gain = (a.lit_gain + b.lit_gain) / 2;
        dst.w_adapt = (a.w_adapt + b.w_adapt) / 2;
    }
    for (std::size_t i = 0; i < dst.o0_probs.size(); ++i) {
        dst.o0_probs[i] = avg16(a.o0_probs[i], b.o0_probs[i]);
    }
}

void run_dp_chunk(const ModelState& price_model, const ModelState& lit_model,
                  const PriceTables& pt, const ParserState& entry_ps,
                  const std::uint8_t* data, std::uint32_t chunk_start,
                  std::uint32_t chunk_len, std::uint32_t n,
                  std::uint32_t nice_len, MatchFinder* mf,
                  std::vector<Match>* cached_matches,
                  std::vector<MatchSlice>* cached_slices,
                  bool collect_matches, std::uint32_t& len_end,
                  std::vector<OptNode>& dp, std::vector<Token>& out_tokens) {
    const std::uint32_t max_possible = n - chunk_start;
    if (collect_matches) {
        len_end = chunk_len;
        cached_matches->clear();
        cached_slices->clear();
    }

    const std::size_t dp_cap = std::min<std::size_t>(
        max_possible + 1u, static_cast<std::size_t>(chunk_len) + kMaxMatchLen * 2u + 8u);
    if (dp.size() < dp_cap) dp.resize(dp_cap);
    for (std::size_t i = 0; i < dp_cap; ++i) {
        dp[i].cost = kInfinity;
    }

    dp[0].cost = 0;
    dp[0].state = entry_ps.state;
    dp[0].rep = entry_ps.rep;

    std::vector<Match> local_matches;
    std::uint32_t i = 0;

    while (i < len_end) {
        const OptNode cur = dp[i];
        if (cur.cost >= kInfinity) {
            if (collect_matches) {
                if (mf) mf->skip(chunk_start + i);
                cached_slices->push_back({static_cast<std::uint32_t>(cached_matches->size()), 0u});
            }
            ++i;
            continue;
        }

        const std::uint32_t pos = chunk_start + i;
        const std::uint32_t pos_state = pos & price_model.pos_mask;
        const std::uint8_t state = cur.state;
        const auto rep = cur.rep;
        const bool can_grow = collect_matches && (i < chunk_len);
        const std::uint32_t max_avail = can_grow
            ? std::min<std::uint32_t>(kMaxMatchLen, max_possible - i)
            : std::min<std::uint32_t>(kMaxMatchLen, len_end - i);

        // Fetch or record matches at `pos`
        const Match* m_ptr = nullptr;
        std::uint32_t m_cnt = 0;
        if (collect_matches) {
            mf->find(pos, local_matches);
            const std::uint32_t off = static_cast<std::uint32_t>(cached_matches->size());
            m_cnt = static_cast<std::uint32_t>(local_matches.size());
            cached_matches->insert(cached_matches->end(),
                                   local_matches.begin(), local_matches.end());
            cached_slices->push_back({off, m_cnt});
            m_ptr = cached_matches->data() + off;
        } else {
            const auto sl = (*cached_slices)[i];
            m_ptr = cached_matches->data() + sl.offset;
            m_cnt = sl.count;
        }

        auto relax = [&](std::uint32_t next_i, std::uint64_t new_cost,
                         std::uint8_t new_state,
                         const std::array<std::uint32_t, 4>& new_rep,
                         std::uint8_t kind, std::uint16_t len,
                         std::uint16_t len2, std::uint32_t dist_or_rep) {
            if (next_i > len_end) {
                if (!can_grow) return;
                len_end = next_i;
                if (dp.size() <= len_end) dp.resize(len_end + 1u);
            }
            if (new_cost < dp[next_i].cost) {
                dp[next_i].cost = new_cost;
                dp[next_i].back = i;
                dp[next_i].state = new_state;
                dp[next_i].rep = new_rep;
                dp[next_i].kind = kind;
                dp[next_i].len = len;
                dp[next_i].len2 = len2;
                dp[next_i].dist_or_rep = dist_or_rep;
            }
        };

        const std::uint8_t prev_b = data[pos - 1];
        const std::uint8_t prev2_b = (pos > 1) ? data[pos - 2] : 0;
        const std::uint8_t prev3_b = (pos > 2) ? data[pos - 3] : 0;
        const std::uint8_t prev4_b = (pos > 3) ? data[pos - 4] : 0;
        const std::uint32_t match_ctx = match_ctx_of(pos, price_model.pos_mask, prev_b);
        const std::uint32_t rep_ctx = rep_ctx_of(pos, price_model.pos_mask, prev_b);
        const std::uint32_t match_hdr = price_model.is_match[state][match_ctx].price(1);
        const std::uint32_t rep_base = match_hdr + price_model.is_rep[state].price(1);

        // Helper for 2-step: Token(i, len1) + Lit(i + len1) + ShortRep/Rep0(i + len1 + 1)
        auto try_token_lit_rep0 = [&](std::uint32_t len1, std::uint64_t tok_cost,
                                      std::uint8_t tok_state,
                                      const std::array<std::uint32_t, 4>& tok_rep,
                                      std::uint8_t compound_kind,
                                      std::uint32_t tok_dist_or_rep) {
            const std::uint32_t lit_pos = pos + len1;
            const std::uint32_t rem = can_grow ? (max_possible - (i + len1)) : (len_end - (i + len1));
            if (rem < 2u) return;
            const std::uint32_t r0 = tok_rep[0];
            if (lit_pos + 1u < r0 || data[lit_pos + 1u] != data[lit_pos + 1u - r0]) {
                return;
            }
            const std::uint8_t l_mb = data[lit_pos - r0];
            const std::uint8_t l_prev3 = (lit_pos > 2u) ? data[lit_pos - 3u] : 0u;
            const std::uint8_t l_prev4 = (lit_pos > 3u) ? data[lit_pos - 4u] : 0u;
            const std::uint32_t l_mctx = match_ctx_of(
                lit_pos, price_model.pos_mask, data[lit_pos - 1]);
            const std::uint32_t l_price =
                price_model.is_match[tok_state][l_mctx].price(0) +
                price_literal(price_model, lit_model, lit_pos, data[lit_pos - 1],
                              data[lit_pos - 2], l_prev3, l_prev4, true, l_mb, data[lit_pos]);
            const std::uint8_t after_l_s = kNextStateLit[tok_state];
            const std::uint32_t r0_pos = lit_pos + 1u;
            const std::uint32_t r0_ps = r0_pos & price_model.pos_mask;
            const std::uint32_t r0_mctx = match_ctx_of(
                r0_pos, price_model.pos_mask, data[lit_pos]);
            const std::uint32_t r0_rctx = rep_ctx_of(
                r0_pos, price_model.pos_mask, data[lit_pos]);
            const std::uint64_t r0_hdr = tok_cost + l_price +
                price_model.is_match[after_l_s][r0_mctx].price(1) +
                price_model.is_rep[after_l_s].price(1) +
                price_model.is_rep0[after_l_s].price(0);

            // Check ShortRep
            if (rem == 2u || data[r0_pos + 1u] != data[r0_pos + 1u - r0]) {
                const std::uint64_t c = r0_hdr +
                    price_model.is_rep0_long[after_l_s][r0_rctx].price(0);
                relax(i + len1 + 2u, c, state_after_shortrep(after_l_s), tok_rep,
                      compound_kind, static_cast<std::uint16_t>(len1), 1u,
                      tok_dist_or_rep);
            } else {
                const std::uint32_t max2 = std::min<std::uint32_t>(kMaxMatchLen, rem - 1u);
                const std::uint32_t len2 = match_len_fast(
                    data + r0_pos - r0, data + r0_pos, 2u, max2);
                const std::uint64_t c = r0_hdr +
                    price_model.is_rep0_long[after_l_s][r0_rctx].price(1) +
                    pt.price_rep_len(r0_ps, len2);
                relax(i + len1 + 1u + len2, c, state_after_rep(after_l_s), tok_rep,
                      compound_kind, static_cast<std::uint16_t>(len1),
                      static_cast<std::uint16_t>(len2), tok_dist_or_rep);
            }
        };

        // 1. Rep0..Rep3
        std::uint32_t max_rep_len = 0u;
        for (std::uint32_t r = 0; r < 4u; ++r) {
            const std::uint32_t d = rep[r];
            if (d > pos || max_avail < 2u) continue;
            if (data[pos] != data[pos - d] || data[pos + 1] != data[pos + 1 - d]) {
                continue;
            }
            const std::uint32_t rlen = match_len_fast(data + pos - d, data + pos, 2u, max_avail);
            if (rlen > max_rep_len) max_rep_len = rlen;

            std::uint64_t hdr = cur.cost + rep_base;
            std::array<std::uint32_t, 4> next_rep = rep;
            if (r == 0u) {
                hdr += price_model.is_rep0[state].price(0) +
                       price_model.is_rep0_long[state][rep_ctx].price(1);
            } else {
                hdr += price_model.is_rep0[state].price(1);
                if (r == 1u) {
                    hdr += price_model.is_rep1[state].price(0);
                    std::swap(next_rep[0], next_rep[1]);
                } else if (r == 2u) {
                    hdr += price_model.is_rep1[state].price(1) +
                           price_model.is_rep2[state].price(0);
                    next_rep[2] = rep[1];
                    next_rep[1] = rep[0];
                    next_rep[0] = d;
                } else {
                    hdr += price_model.is_rep1[state].price(1) +
                           price_model.is_rep2[state].price(1);
                    next_rep[3] = rep[2];
                    next_rep[2] = rep[1];
                    next_rep[1] = rep[0];
                    next_rep[0] = d;
                }
            }

            const std::uint8_t next_s = state_after_rep(state);
            auto eval_rep_len = [&](std::uint32_t l) {
                relax(i + l, hdr + pt.price_rep_len(pos_state, l), next_s, next_rep,
                      kOptRep, static_cast<std::uint16_t>(l), 0u, r);
            };

            if (rlen >= nice_len) {
                eval_rep_len(rlen - 1u);
                eval_rep_len(rlen);
            } else {
                const std::uint32_t dense_end = std::min<std::uint32_t>(rlen, 24u);
                for (std::uint32_t l = 2u; l <= dense_end; ++l) {
                    eval_rep_len(l);
                }
                if (rlen > 24u) {
                    static constexpr std::uint32_t kKeyLens[] = {
                        32, 48, 64, 96, 128, 192
                    };
                    for (std::uint32_t kl : kKeyLens) {
                        if (kl < rlen - 2u) eval_rep_len(kl);
                    }
                    for (std::uint32_t l = std::max<std::uint32_t>(25u, rlen - 2u); l <= rlen; ++l) {
                        eval_rep_len(l);
                    }
                }
            }
            try_token_lit_rep0(rlen, hdr + pt.price_rep_len(pos_state, rlen),
                               next_s, next_rep, kOptRepLitRep0, r);
        }

        // 2. Normal Matches (Pareto candidates from Hash2/Hash3/Hash6/Hash8/BT4, len >= 3)
        std::uint32_t max_norm_len = 0u;
        if (m_cnt > 0u && max_avail >= 3u) {
            const std::uint64_t norm_hdr = cur.cost + match_hdr +
                                           price_model.is_rep[state].price(0);
            const std::uint8_t next_s = state_after_match(state);
            std::uint32_t start_l = 3u;

            for (std::uint32_t mi = 0; mi < m_cnt; ++mi) {
                const std::uint32_t m_dist = m_ptr[mi].dist;
                const std::uint32_t m_len = std::min(m_ptr[mi].len, max_avail);
                if (m_dist > pos || m_len < start_l) continue;
                if (m_len > max_norm_len) max_norm_len = m_len;

                std::array<std::uint32_t, 4> next_rep = {m_dist, rep[0], rep[1], rep[2]};
                auto eval_match_len = [&](std::uint32_t l) {
                    const std::uint64_t c = norm_hdr +
                        pt.price_match_len(pos_state, l) +
                        pt.price_dist(l, m_dist, pos);
                    relax(i + l, c, next_s, next_rep, kOptMatch,
                          static_cast<std::uint16_t>(l), 0u, m_dist);
                };

                if (m_len >= nice_len) {
                    if (m_len - 1u >= start_l) eval_match_len(m_len - 1u);
                    eval_match_len(m_len);
                } else {
                    const std::uint32_t dense_end = std::min<std::uint32_t>(m_len, 24u);
                    for (std::uint32_t l = start_l; l <= dense_end; ++l) {
                        eval_match_len(l);
                    }
                    if (m_len > 24u) {
                        static constexpr std::uint32_t kKeyLens[] = {
                            32, 48, 64, 96, 128, 192
                        };
                        for (std::uint32_t kl : kKeyLens) {
                            if (kl >= start_l && kl < m_len - 2u) eval_match_len(kl);
                        }
                        for (std::uint32_t l = std::max<std::uint32_t>(
                                 std::max<std::uint32_t>(start_l, 25u), m_len - 2u);
                             l <= m_len; ++l) {
                            eval_match_len(l);
                        }
                    }
                }

                if (m_len >= 3u) {
                    const std::uint64_t end_c = norm_hdr +
                        pt.price_match_len(pos_state, m_len) +
                        pt.price_dist(m_len, m_dist, pos);
                    try_token_lit_rep0(m_len, end_c, next_s, next_rep,
                                       kOptMatchLitRep0, m_dist);
                }
                start_l = m_len + 1u;
            }
        }

        // 3. ShortRep & Literal at `i` (skipped when a >= nice_len match allows fast-forwarding)
        if (max_rep_len < nice_len && max_norm_len < nice_len) {
            if (pos >= rep[0] && data[pos] == data[pos - rep[0]]) {
                const std::uint64_t sr_cost = cur.cost + rep_base +
                    price_model.is_rep0[state].price(0) +
                    price_model.is_rep0_long[state][rep_ctx].price(0);
                relax(i + 1u, sr_cost, state_after_shortrep(state), rep,
                      kOptShortRep, 1u, 0u, 0u);
                try_token_lit_rep0(1u, sr_cost, state_after_shortrep(state), rep,
                                   kOptRepLitRep0, 0u);
            }

            const bool matched = (state >= 7);
            const std::uint8_t mb = matched ? data[pos - rep[0]] : 0;
            const std::uint32_t lit_price =
                price_model.is_match[state][match_ctx].price(0) +
                price_literal(price_model, lit_model, pos, prev_b, prev2_b,
                              prev3_b, prev4_b, matched, mb, data[pos]);
            const std::uint8_t lit_state = kNextStateLit[state];
            const std::uint64_t lit_end_cost = cur.cost + lit_price;
            relax(i + 1u, lit_end_cost, lit_state, rep, kOptLit, 1u, 0u, 0u);

            // 2-step compound: Literal(i) + ShortRep/Rep0(i + 1)
            const std::uint32_t avail_after_lit = can_grow
                ? std::min<std::uint32_t>(kMaxMatchLen, max_possible - (i + 1u))
                : (len_end - (i + 1u));
            if (avail_after_lit >= 1u && pos + 1u >= rep[0] &&
                data[pos + 1u] == data[pos + 1u - rep[0]]) {
                const std::uint32_t p1 = pos + 1u;
                const std::uint32_t ps1 = p1 & price_model.pos_mask;
                const std::uint32_t mctx1 = match_ctx_of(p1, price_model.pos_mask, data[pos]);
                const std::uint32_t rctx1 = rep_ctx_of(p1, price_model.pos_mask, data[pos]);
                const std::uint64_t base_r0 = lit_end_cost +
                    price_model.is_match[lit_state][mctx1].price(1) +
                    price_model.is_rep[lit_state].price(1) +
                    price_model.is_rep0[lit_state].price(0);

                // ShortRep at i+1
                const std::uint64_t sr_cost = base_r0 +
                    price_model.is_rep0_long[lit_state][rctx1].price(0);
                relax(i + 2u, sr_cost, state_after_shortrep(lit_state), rep,
                      kOptLitRep0, 1u, 1u, 0u);

                // Rep0(len2 >= 2) at i+1
                if (avail_after_lit >= 2u &&
                    data[p1 + 1u] == data[p1 + 1u - rep[0]]) {
                    const std::uint32_t r2 = match_len_fast(
                        data + p1 - rep[0], data + p1, 2u,
                        std::min<std::uint32_t>(kMaxMatchLen, avail_after_lit));
                    const std::uint64_t rep_hdr = base_r0 +
                        price_model.is_rep0_long[lit_state][rctx1].price(1);
                    const std::uint8_t next_s = state_after_rep(lit_state);
                    relax(i + 1u + r2, rep_hdr + pt.price_rep_len(ps1, r2),
                          next_s, rep, kOptLitRep0, static_cast<std::uint16_t>(r2),
                          static_cast<std::uint16_t>(r2), 0u);
                }
            }
        }

        ++i;
    }

    // Backtrack from `dp[len_end]` to produce tokens in forward order.
    out_tokens.clear();
    std::vector<Token> rev;
    std::uint32_t cur_i = len_end;
    while (cur_i > 0u) {
        const OptNode& nd = dp[cur_i];
        switch (nd.kind) {
            case kOptLit:
                rev.push_back({kTokLit, 1u, 0u});
                break;
            case kOptShortRep:
                rev.push_back({kTokShortRep, 1u, 0u});
                break;
            case kOptRep:
                rev.push_back({kTokRep, nd.len, nd.dist_or_rep});
                break;
            case kOptMatch:
                rev.push_back({kTokMatch, nd.len, nd.dist_or_rep});
                break;
            case kOptLitRep0:
                rev.push_back({nd.len2 == 1u ? kTokShortRep : kTokRep, nd.len2, 0u});
                rev.push_back({kTokLit, 1u, 0u});
                break;
            case kOptMatchLitRep0:
                rev.push_back({nd.len2 == 1u ? kTokShortRep : kTokRep, nd.len2, 0u});
                rev.push_back({kTokLit, 1u, 0u});
                rev.push_back({kTokMatch, nd.len, nd.dist_or_rep});
                break;
            case kOptRepLitRep0:
                rev.push_back({nd.len2 == 1u ? kTokShortRep : kTokRep, nd.len2, 0u});
                rev.push_back({kTokLit, 1u, 0u});
                rev.push_back({nd.len == 1u ? kTokShortRep : kTokRep, nd.len, nd.dist_or_rep});
                break;
        }
        cur_i = nd.back;
    }
    out_tokens.assign(rev.rbegin(), rev.rend());
}

void compress_optimal(RangeEncoder& rc, ModelState& m, MatchFinder& mf,
                      const std::uint8_t* data, std::uint32_t n, int level) {
    if (n == 0u) return;

    ParserState ps{};
    emit_token(rc, m, ps, data, 0u, {kTokLit, 1u, 0u});
    mf.skip(0u);
    if (n == 1u) return;

    const bool two_pass = (level >= 10);
    const bool three_pass = (level >= 12);
    const std::uint32_t nice_len =
        (n <= 524288u) ? (kMaxMatchLen + 1u) : ((level >= 11) ? 128u : 48u);
    PriceTables pt;
    std::vector<OptNode> dp(kOptChunk + kMaxMatchLen * 2u + 8u);
    std::vector<Match> cached_matches;
    cached_matches.reserve(kOptChunk * 6u);
    std::vector<MatchSlice> cached_slices;
    cached_slices.reserve(kOptChunk + kMaxMatchLen * 2u + 8u);
    std::vector<Token> tokens1;
    std::vector<Token> tokens2;
    std::vector<Token> tokens3;
    tokens1.reserve(kOptChunk);
    tokens2.reserve(kOptChunk);
    tokens3.reserve(kOptChunk);
    std::vector<ProbUndo> undo_log;
    undo_log.reserve(24576u);

    ModelState sim_m1;
    ModelState sim_m2;
    ModelState sim_m3;
    ModelState blended_m;
    if (two_pass) {
        sim_m1.init(m.lit_mode, m.pb, n, /*alloc_lit_tables=*/false);
        sim_m2.init(m.lit_mode, m.pb, n, /*alloc_lit_tables=*/false);
        if (three_pass) sim_m3.init(m.lit_mode, m.pb, n, /*alloc_lit_tables=*/false);
        blended_m.init(m.lit_mode, m.pb, n, /*alloc_lit_tables=*/false);
    }

    std::uint32_t pos = 1u;
    while (pos < n) {
        // Finer warmup chunks refresh PriceTables and model probabilities more often.
        const std::uint32_t target_chunk =
            (pos < 1024u && n > 512u) ? 256u
            : ((pos < 4096u && n > 2048u) ? 512u
            : ((pos < 16384u && n > 4096u) ? 1024u
                                           : static_cast<std::uint32_t>(kOptChunk)));
        const std::uint32_t chunk_len = std::min(target_chunk, n - pos);

        pt.update(m);
        std::uint32_t len_end = chunk_len;
        run_dp_chunk(m, m, pt, ps, data, pos, chunk_len, n, nice_len, &mf,
                     &cached_matches, &cached_slices, /*collect_matches=*/true,
                     len_end, dp, tokens1);

        const std::vector<Token>* chosen = &tokens1;
        if (two_pass && tokens1.size() >= 8u) {
            copy_core_model(sim_m1, m);
            const std::uint64_t cost1 =
                simulate_chunk(sim_m1, m, ps, data, pos, tokens1, undo_log);
            std::uint64_t best_cost = cost1;

            blend_core_models(blended_m, m, sim_m1, /*weight_b=*/1);
            pt.update(blended_m);
            run_dp_chunk(blended_m, m, pt, ps, data, pos, len_end, n, nice_len,
                         nullptr, &cached_matches, &cached_slices,
                         /*collect_matches=*/false, len_end, dp, tokens2);

            copy_core_model(sim_m2, m);
            const std::uint64_t cost2 =
                simulate_chunk(sim_m2, m, ps, data, pos, tokens2, undo_log);
            if (cost2 < best_cost) {
                best_cost = cost2;
                chosen = &tokens2;
            }

            if (three_pass) {
                const ModelState& best_end_m = (cost2 < cost1) ? sim_m2 : sim_m1;
                blend_core_models(blended_m, m, best_end_m, /*weight_b=*/3);
                pt.update(blended_m);
                run_dp_chunk(blended_m, m, pt, ps, data, pos, len_end, n, nice_len,
                             nullptr, &cached_matches, &cached_slices,
                             /*collect_matches=*/false, len_end, dp, tokens3);

                copy_core_model(sim_m3, m);
                const std::uint64_t cost3 =
                    simulate_chunk(sim_m3, m, ps, data, pos, tokens3, undo_log);
                if (cost3 < best_cost) {
                    chosen = &tokens3;
                }
            }
        }

        for (const auto& tok : *chosen) {
            emit_token(rc, m, ps, data, pos, tok);
            pos += tok.len;
        }
    }
}

// ---------------------------------------------------------------------------
// Reversible 1-Byte Dictionary + Large-Table LZP + SA-IS BWT + Rich BAC
// ---------------------------------------------------------------------------
constexpr std::uint32_t kMinLzpLen = 12;
constexpr std::uint32_t kShortLzpSpan = 254;  // codes 1..254 -> lens 12..265
constexpr std::uint32_t kMaxLzpLen = 16384;

inline constexpr std::array<std::string_view, 46> kDictTokens = {
    "</text>\n    </revision>\n  </page>\n  <page>\n    <title>",
    "</timestamp>\n      <contributor>\n        <username>",
    "</comment>\n      <text xml:space=\"preserve\">",
    "</id>\n      </contributor>\n      ",
    "</id>\n    <revision>\n      <id>",
    "<text xml:space=\"preserve\">",
    "</id>\n      <timestamp>200",
    "</username>\n        <id>",
    "==External links==",
    "==References==",
    "</title>\n    <id>",
    "==See also==",
    "#REDIRECT [[",
    "[[Category:",
    "&amp;ndash;",
    "&amp;mdash;",
    "&amp;nbsp;",
    "[[Image:",
    "https://",
    "http://",
    "&quot;",
    " which",
    " their",
    " would",
    " about",
    " there",
    " other",
    "&amp;",
    " that",
    " with",
    " from",
    " have",
    " this",
    " were",
    "&lt;",
    "&gt;",
    " the",
    " and",
    " for",
    " was",
    " are",
    " of",
    " in",
    " to",
    " is",
    " on",
};

/// Reversible pre-filter replacing frequent tokens with unused single bytes.
auto dict_forward(const std::uint8_t* raw, std::uint32_t n,
                  std::vector<std::uint8_t>& syms_out,
                  std::vector<std::uint8_t>& out) -> bool {
    if (n < 65536u) return false;
    std::array<std::uint32_t, 256> freq{};
    for (std::uint32_t i = 0; i < n; ++i) ++freq[raw[i]];

    std::vector<std::uint8_t> unused;
    unused.reserve(64u);
    for (std::uint32_t b = 0; b < 256u; ++b) {
        if (freq[b] == 0u) unused.push_back(static_cast<std::uint8_t>(b));
    }
    // Keep at least one unused byte available as LZP escape symbol.
    if (unused.size() < 8u) return false;

    const std::size_t k = std::min<std::size_t>(unused.size() - 1u, kDictTokens.size());
    syms_out.assign(unused.begin(), unused.begin() + static_cast<std::ptrdiff_t>(k));

    std::vector<std::int8_t> head(65536u, -1);
    std::array<std::int8_t, kDictTokens.size()> next{};
    next.fill(-1);
    for (int j = static_cast<int>(k) - 1; j >= 0; --j) {
        const auto p16 = load_u16(reinterpret_cast<const std::uint8_t*>(
            kDictTokens[static_cast<std::size_t>(j)].data()));
        next[static_cast<std::size_t>(j)] = head[p16];
        head[p16] = static_cast<std::int8_t>(j);
    }

    out.clear();
    out.reserve(n);
    std::uint32_t i = 0;
    while (i < n) {
        if (i + 3u <= n) {
            int j = head[load_u16(raw + i)];
            bool matched = false;
            while (j >= 0) {
                const auto sv = kDictTokens[static_cast<std::size_t>(j)];
                if (i + sv.size() <= n && std::memcmp(raw + i, sv.data(), sv.size()) == 0) {
                    out.push_back(syms_out[static_cast<std::size_t>(j)]);
                    i += static_cast<std::uint32_t>(sv.size());
                    matched = true;
                    break;
                }
                j = next[static_cast<std::size_t>(j)];
            }
            if (matched) continue;
        }
        out.push_back(raw[i++]);
    }
    return out.size() + 256u < n;
}

/// Restores original tokens from unused-byte dictionary codes.
void dict_inverse(const std::uint8_t* in, std::uint32_t in_len,
                  const std::uint8_t* syms, std::size_t k,
                  std::uint8_t* dst, std::uint32_t n) {
    std::array<std::int8_t, 256> sym_to_idx{};
    sym_to_idx.fill(-1);
    for (std::size_t j = 0; j < k; ++j) {
        sym_to_idx[syms[j]] = static_cast<std::int8_t>(j);
    }

    std::uint32_t out_pos = 0;
    for (std::uint32_t in_pos = 0; in_pos < in_len; ++in_pos) {
        const std::uint8_t b = in[in_pos];
        const int j = sym_to_idx[b];
        if (j < 0) {
            if (out_pos >= n) {
                throw std::runtime_error("FZAC dict_inverse overflow");
            }
            dst[out_pos++] = b;
        } else {
            const auto sv = kDictTokens[static_cast<std::size_t>(j)];
            if (sv.size() > n - out_pos) {
                throw std::runtime_error("FZAC dict_inverse token overflow");
            }
            std::memcpy(dst + out_pos, sv.data(), sv.size());
            out_pos += static_cast<std::uint32_t>(sv.size());
        }
    }
    if (out_pos != n) {
        throw std::runtime_error("FZAC dict_inverse size mismatch");
    }
}

inline auto lzp_hash_bits(std::uint32_t n) -> std::uint32_t {
    if (n >= (16u << 20)) return 22u;
    if (n >= (4u << 20)) return 21u;
    if (n >= (1u << 20)) return 19u;
    return 18u;
}

inline auto lzp_hash(const std::uint8_t* p, std::uint32_t hshift) -> std::uint32_t {
    return (load_u32(p - 4) * 2654435761u) >> hshift;
}

auto lzp_forward(const std::uint8_t* raw, std::uint32_t n,
                 std::uint8_t& out_esc, std::vector<std::uint8_t>& out) -> bool {
    if (n < 32768u) return false;
    std::array<std::uint32_t, 256> freq{};
    for (std::uint32_t i = 0; i < n; ++i) ++freq[raw[i]];
    const std::uint8_t esc = static_cast<std::uint8_t>(
        std::min_element(freq.begin(), freq.end()) - freq.begin());
    out_esc = esc;

    const std::uint32_t hbits = lzp_hash_bits(n);
    const std::uint32_t hshift = 32u - hbits;
    std::vector<std::uint32_t> lut(std::size_t{1} << hbits, kNullPos);

    out.clear();
    out.reserve(n);
    std::uint32_t i = 0;
    while (i < n) {
        std::uint32_t ref = kNullPos;
        if (i >= 4u) {
            const std::uint32_t h = lzp_hash(raw + i, hshift);
            ref = lut[h];
            lut[h] = i;
        }
        const std::uint32_t limit = std::min<std::uint32_t>(kMaxLzpLen, n - i);
        if (ref != kNullPos && limit >= kMinLzpLen &&
            load_u64(raw + ref) == load_u64(raw + i) &&
            load_u32(raw + ref + 8u) == load_u32(raw + i + 8u)) {
            const std::uint32_t mlen =
                match_len_fast(raw + ref, raw + i, kMinLzpLen, limit);
            out.push_back(esc);
            if (mlen < kMinLzpLen + kShortLzpSpan) {
                out.push_back(static_cast<std::uint8_t>((mlen - kMinLzpLen) + 1u));
            } else {
                out.push_back(255u);
                std::uint32_t extra = mlen - (kMinLzpLen + kShortLzpSpan);
                while (extra >= 0x80u) {
                    out.push_back(static_cast<std::uint8_t>((extra & 0x7Fu) | 0x80u));
                    extra >>= 7;
                }
                out.push_back(static_cast<std::uint8_t>(extra));
            }
            i += mlen;
            continue;
        }
        const std::uint8_t b = raw[i++];
        out.push_back(b);
        if (b == esc) out.push_back(0u);
    }
    return out.size() + 64u < n;
}

void lzp_inverse(const std::uint8_t* in, std::uint32_t in_len,
                 std::uint8_t esc, std::uint8_t* dst, std::uint32_t n) {
    const std::uint32_t hbits = lzp_hash_bits(n);
    const std::uint32_t hshift = 32u - hbits;
    std::vector<std::uint32_t> lut(std::size_t{1} << hbits, kNullPos);
    std::uint32_t in_pos = 0;
    std::uint32_t out_pos = 0;
    while (out_pos < n) {
        if (in_pos >= in_len) {
            throw std::runtime_error("truncated FZAC LZP stream");
        }
        std::uint32_t ref = kNullPos;
        if (out_pos >= 4u) {
            const std::uint32_t h = lzp_hash(dst + out_pos, hshift);
            ref = lut[h];
            lut[h] = out_pos;
        }
        const std::uint8_t b = in[in_pos++];
        if (b != esc) {
            dst[out_pos++] = b;
        } else {
            if (in_pos >= in_len) {
                throw std::runtime_error("truncated FZAC LZP escape");
            }
            const std::uint8_t code = in[in_pos++];
            if (code == 0u) {
                dst[out_pos++] = esc;
            } else {
                std::uint32_t mlen = 0u;
                if (code < 255u) {
                    mlen = static_cast<std::uint32_t>(code - 1u) + kMinLzpLen;
                } else {
                    std::uint32_t extra = 0u;
                    std::uint32_t shift = 0u;
                    for (;;) {
                        if (in_pos >= in_len || shift >= 28u) {
                            throw std::runtime_error("corrupted FZAC LZP extended length");
                        }
                        const std::uint8_t vb = in[in_pos++];
                        extra |= static_cast<std::uint32_t>(vb & 0x7Fu) << shift;
                        if ((vb & 0x80u) == 0u) break;
                        shift += 7u;
                    }
                    mlen = kMinLzpLen + kShortLzpSpan + extra;
                }
                if (ref == kNullPos || ref >= out_pos || out_pos + mlen > n) {
                    throw std::runtime_error("corrupted FZAC LZP match");
                }
                std::uint8_t* wp = dst + out_pos;
                const std::uint8_t* rp = dst + ref;
                if (out_pos - ref >= mlen) {
                    std::memcpy(wp, rp, mlen);
                } else {
                    for (std::uint32_t k = 0; k < mlen; ++k) wp[k] = rp[k];
                }
                out_pos += mlen;
            }
        }
    }
    if (in_pos != in_len) {
        throw std::runtime_error("corrupted FZAC LZP stream length");
    }
}

void sais_induce(const std::int32_t* T, std::int32_t* SA,
                 const std::uint8_t* t, const std::int32_t* bucket_sizes,
                 std::int32_t* b, std::int32_t N, std::int32_t K) {
    std::int32_t sum = 0;
    for (std::int32_t c = 0; c < K; ++c) {
        b[c] = sum;
        sum += bucket_sizes[c];
    }
    for (std::int32_t i = 0; i < N; ++i) {
        const std::int32_t j = SA[i] - 1;
        if (j >= 0 && t[j] == 0u) {
            SA[b[T[j]]++] = j;
        }
    }
    sum = 0;
    for (std::int32_t c = 0; c < K; ++c) {
        sum += bucket_sizes[c];
        b[c] = sum - 1;
    }
    for (std::int32_t i = N - 1; i >= 0; --i) {
        const std::int32_t j = SA[i] - 1;
        if (j >= 0 && t[j] != 0u) {
            SA[b[T[j]]--] = j;
        }
    }
}

void sais_core(const std::int32_t* T, std::int32_t* SA,
               std::int32_t N, std::int32_t K) {
    std::vector<std::uint8_t> t(static_cast<std::size_t>(N), 0u);
    t[N - 1] = 1u;
    for (std::int32_t i = N - 2; i >= 0; --i) {
        if (T[i] < T[i + 1] || (T[i] == T[i + 1] && t[i + 1] != 0u)) {
            t[i] = 1u;
        }
    }
    auto is_lms = [&](std::int32_t i) -> bool {
        return i > 0 && t[i] != 0u && t[i - 1] == 0u;
    };

    std::vector<std::int32_t> bucket_sizes(static_cast<std::size_t>(K), 0);
    std::vector<std::int32_t> b(static_cast<std::size_t>(K), 0);
    for (std::int32_t i = 0; i < N; ++i) ++bucket_sizes[T[i]];

    std::fill(SA, SA + N, -1);
    std::int32_t sum = 0;
    for (std::int32_t c = 0; c < K; ++c) {
        sum += bucket_sizes[c];
        b[c] = sum - 1;
    }
    for (std::int32_t i = 1; i < N; ++i) {
        if (is_lms(i)) SA[b[T[i]]--] = i;
    }
    sais_induce(T, SA, t.data(), bucket_sizes.data(), b.data(), N, K);

    std::int32_t n1 = 0;
    for (std::int32_t i = 0; i < N; ++i) {
        if (is_lms(SA[i])) SA[n1++] = SA[i];
    }
    std::fill(SA + n1, SA + N, -1);

    std::int32_t name = 0;
    std::int32_t prev = -1;
    for (std::int32_t i = 0; i < n1; ++i) {
        const std::int32_t pos = SA[i];
        bool diff = false;
        if (prev == -1) {
            diff = true;
        } else {
            for (std::int32_t d = 0; ; ++d) {
                if (T[pos + d] != T[prev + d] || t[pos + d] != t[prev + d]) {
                    diff = true;
                    break;
                }
                if (d > 0 && (is_lms(pos + d) || is_lms(prev + d))) {
                    if (!(is_lms(pos + d) && is_lms(prev + d))) diff = true;
                    break;
                }
            }
        }
        if (diff) {
            ++name;
            prev = pos;
        }
        SA[n1 + (pos >> 1)] = name - 1;
    }

    std::int32_t* S1 = SA + N - n1;
    for (std::int32_t i = N - 1, j = N - 1; i >= n1; --i) {
        if (SA[i] >= 0) SA[j--] = SA[i];
    }

    std::int32_t* SA1 = SA;
    if (name < n1) {
        sais_core(S1, SA1, n1, name);
    } else {
        for (std::int32_t i = 0; i < n1; ++i) SA1[S1[i]] = i;
    }

    for (std::int32_t i = 1, j = 0; i < N; ++i) {
        if (is_lms(i)) S1[j++] = i;
    }
    for (std::int32_t i = 0; i < n1; ++i) SA1[i] = S1[SA1[i]];
    std::fill(SA + n1, SA + N, -1);

    sum = 0;
    for (std::int32_t c = 0; c < K; ++c) {
        sum += bucket_sizes[c];
        b[c] = sum - 1;
    }
    for (std::int32_t i = n1 - 1; i >= 0; --i) {
        const std::int32_t j = SA1[i];
        SA1[i] = -1;
        SA[b[T[j]]--] = j;
    }
    sais_induce(T, SA, t.data(), bucket_sizes.data(), b.data(), N, K);
}

void bwt_forward(const std::uint8_t* buf, std::uint32_t m,
                 std::vector<std::uint8_t>& L, std::uint32_t& primary_idx) {
    L.resize(m);
    if (m == 0u) {
        primary_idx = 0u;
        return;
    }
    std::vector<std::int32_t> T(static_cast<std::size_t>(m) + 1u);
    for (std::uint32_t i = 0; i < m; ++i) {
        T[i] = static_cast<std::int32_t>(buf[i]) + 1;
    }
    T[m] = 0;
    std::vector<std::int32_t> SA(static_cast<std::size_t>(m) + 1u);
    sais_core(T.data(), SA.data(), static_cast<std::int32_t>(m + 1u), 257);

    std::uint32_t out_i = 0;
    primary_idx = 0;
    for (std::uint32_t r = 0; r <= m; ++r) {
        if (SA[r] == 0) {
            primary_idx = r - 1u;
        } else {
            L[out_i++] = buf[SA[r] - 1];
        }
    }
}

void bwt_inverse(const std::uint8_t* L, std::uint32_t m,
                 std::uint32_t primary_idx, std::uint8_t* dst) {
    if (m == 0u) return;
    if (primary_idx >= m) {
        throw std::runtime_error("invalid FZAC BWT primary index");
    }
    std::array<std::uint32_t, 256> cumul{};
    for (std::uint32_t i = 0; i < m; ++i) ++cumul[L[i]];
    std::uint32_t sum = 0;
    for (std::uint32_t c = 0; c < 256u; ++c) {
        const std::uint32_t cnt = cumul[c];
        cumul[c] = sum;
        sum += cnt;
    }
    std::vector<std::uint32_t> lf(m);
    if (m <= (1u << 24)) {
        for (std::uint32_t i = 0; i < m; ++i) {
            const std::uint8_t c = L[i];
            const std::uint32_t rank = cumul[c]++;
            const std::uint32_t next_idx =
                (rank < primary_idx) ? (rank + 1u)
                                     : ((rank == primary_idx) ? 0u : rank);
            lf[i] = (next_idx << 8) | static_cast<std::uint32_t>(c);
        }
        std::uint32_t idx = 0;
        for (std::uint32_t k = m; k > 0u; --k) {
            const std::uint32_t entry = lf[idx];
            dst[k - 1u] = static_cast<std::uint8_t>(entry & 0xFFu);
            idx = entry >> 8;
        }
    } else {
        for (std::uint32_t i = 0; i < m; ++i) {
            const std::uint8_t c = L[i];
            const std::uint32_t rank = cumul[c]++;
            lf[i] = (rank < primary_idx)
                        ? (rank + 1u)
                        : ((rank == primary_idx) ? 0u : rank);
        }
        std::uint32_t idx = 0;
        for (std::uint32_t k = m; k > 0u; --k) {
            dst[k - 1u] = L[idx];
            idx = lf[idx];
        }
    }
}

/// Multi-context zero-run length coder for MTF zero runs.
struct ZeroRunCoder {
    std::array<std::uint16_t, 2048> gt1_0{};
    std::array<std::uint16_t, 2048> gt2_0{};
    std::array<std::uint16_t, 2048> gt3_0{};
    std::array<std::uint16_t, 2048> ge12_0{};

    std::array<std::uint16_t, 1024> gt1_1{};
    std::array<std::uint16_t, 1024> gt2_1{};
    std::array<std::uint16_t, 1024> gt3_1{};
    std::array<std::uint16_t, 1024> ge12_1{};

    std::array<std::uint16_t, 32> gt1_s{};
    std::array<std::uint16_t, 32> gt2_s{};
    std::array<std::uint16_t, 32> gt3_s{};
    std::array<std::uint16_t, 32> ge12_s{};

    std::array<std::array<std::uint16_t, 8>, 2048> low_0{};
    std::array<std::array<std::uint16_t, 8>, 32> low_s{};
    std::array<std::array<std::uint16_t, 32>, 32> exp_s{};
    std::array<std::uint16_t, 32> exp_g{};
    std::array<std::array<std::array<std::uint16_t, 8>, 32>, 32> mant_s{};
    std::array<std::array<std::uint16_t, 8>, 32> mant_g{};

    void init() {
        gt1_0.fill(kProbInit);
        gt2_0.fill(kProbInit);
        gt3_0.fill(kProbInit);
        ge12_0.fill(kProbInit);
        gt1_1.fill(kProbInit);
        gt2_1.fill(kProbInit);
        gt3_1.fill(kProbInit);
        ge12_1.fill(kProbInit);
        gt1_s.fill(kProbInit);
        gt2_s.fill(kProbInit);
        gt3_s.fill(kProbInit);
        ge12_s.fill(kProbInit);
        for (auto& r : low_0) r.fill(kProbInit);
        for (auto& r : low_s) r.fill(kProbInit);
        for (auto& r : exp_s) r.fill(kProbInit);
        exp_g.fill(kProbInit);
        for (auto& m : mant_s) {
            for (auto& r : m) r.fill(kProbInit);
        }
        for (auto& r : mant_g) r.fill(kProbInit);
    }

    static auto tri_prob(std::uint16_t p0, std::uint16_t p1,
                         std::uint16_t ps) -> std::uint32_t {
        const std::uint32_t e0 = (p0 == kProbInit) ? ps : p0;
        const std::uint32_t e1 = (p1 == kProbInit) ? ps : p1;
        return (2u * e0 + e1 + ps + 2u) >> 2;
    }

    static void encode_tri(RangeEncoder& rc, std::uint16_t& p0,
                           std::uint16_t& p1, std::uint16_t& ps, int bit) {
        rc.encode_bit_prob(tri_prob(p0, p1, ps), bit);
        update_prob<4>(p0, bit);
        update_prob<4>(p1, bit);
        update_prob<3>(ps, bit);
    }

    static auto decode_tri(RangeDecoder& rd, std::uint16_t& p0,
                           std::uint16_t& p1, std::uint16_t& ps) -> int {
        const int bit = rd.decode_bit_prob(tri_prob(p0, p1, ps));
        update_prob<4>(p0, bit);
        update_prob<4>(p1, bit);
        update_prob<3>(ps, bit);
        return bit;
    }

    static void encode_duo(RangeEncoder& rc, std::uint16_t& ps,
                           std::uint16_t& pg, int bit) {
        rc.encode_bit_prob(mix2_backoff(pg, ps), bit);
        update_prob<3>(ps, bit);
        update_prob<4>(pg, bit);
    }

    static auto decode_duo(RangeDecoder& rd, std::uint16_t& ps,
                           std::uint16_t& pg) -> int {
        const int bit = rd.decode_bit_prob(mix2_backoff(pg, ps));
        update_prob<3>(ps, bit);
        update_prob<4>(pg, bit);
        return bit;
    }

    void encode(RangeEncoder& rc, std::uint8_t c0, std::uint32_t prev_rank_b,
                std::uint32_t prev2_rank_b, std::uint32_t last_zrun_b,
                std::uint32_t zrun) {
        const std::uint32_t zctx0 = (static_cast<std::uint32_t>(c0) << 3) | prev_rank_b;
        const std::uint32_t zctx1 = (prev_byte_class(c0) << 8) | (last_zrun_b << 6) |
                                    (prev_rank_b << 3) | prev2_rank_b;
        const std::uint32_t zctx_s = (prev_byte_class(c0) << 3) | prev_rank_b;

        if (zrun == 1u) {
            encode_tri(rc, gt1_0[zctx0], gt1_1[zctx1], gt1_s[zctx_s], 0);
            return;
        }
        encode_tri(rc, gt1_0[zctx0], gt1_1[zctx1], gt1_s[zctx_s], 1);
        if (zrun == 2u) {
            encode_tri(rc, gt2_0[zctx0], gt2_1[zctx1], gt2_s[zctx_s], 0);
            return;
        }
        encode_tri(rc, gt2_0[zctx0], gt2_1[zctx1], gt2_s[zctx_s], 1);
        if (zrun == 3u) {
            encode_tri(rc, gt3_0[zctx0], gt3_1[zctx1], gt3_s[zctx_s], 0);
            return;
        }
        encode_tri(rc, gt3_0[zctx0], gt3_1[zctx1], gt3_s[zctx_s], 1);

        if (zrun <= 11u) {
            encode_tri(rc, ge12_0[zctx0], ge12_1[zctx1], ge12_s[zctx_s], 0);
            const std::uint32_t v = zrun - 4u;
            std::uint32_t tm = 1u;
            for (int i = 2; i >= 0; --i) {
                const int bit = static_cast<int>((v >> i) & 1u);
                rc.encode_bit_prob(mix2_backoff(low_s[zctx_s][tm], low_0[zctx0][tm]), bit);
                update_prob<4>(low_0[zctx0][tm], bit);
                update_prob<3>(low_s[zctx_s][tm], bit);
                tm = (tm << 1) | static_cast<std::uint32_t>(bit);
            }
            return;
        }

        encode_tri(rc, ge12_0[zctx0], ge12_1[zctx1], ge12_s[zctx_s], 1);
        const std::uint32_t u = zrun - 11u;
        const int k = static_cast<int>(std::bit_width(u)) - 1;
        for (int i = 0; i < k; ++i) {
            const int b = std::min(i, 31);
            encode_duo(rc, exp_s[zctx_s][b], exp_g[b], 1);
        }
        {
            const int bk = std::min(k, 31);
            encode_duo(rc, exp_s[zctx_s][bk], exp_g[bk], 0);
        }
        const int bk = std::min(k, 31);
        for (int i = k - 1; i >= 0; --i) {
            const int bpos = std::min(k - 1 - i, 7);
            encode_duo(rc, mant_s[zctx_s][bk][bpos], mant_g[bk][bpos],
                       static_cast<int>((u >> i) & 1u));
        }
    }

    auto decode(RangeDecoder& rd, std::uint8_t c0, std::uint32_t prev_rank_b,
                std::uint32_t prev2_rank_b, std::uint32_t last_zrun_b) -> std::uint32_t {
        const std::uint32_t zctx0 = (static_cast<std::uint32_t>(c0) << 3) | prev_rank_b;
        const std::uint32_t zctx1 = (prev_byte_class(c0) << 8) | (last_zrun_b << 6) |
                                    (prev_rank_b << 3) | prev2_rank_b;
        const std::uint32_t zctx_s = (prev_byte_class(c0) << 3) | prev_rank_b;

        if (decode_tri(rd, gt1_0[zctx0], gt1_1[zctx1], gt1_s[zctx_s]) == 0) return 1u;
        if (decode_tri(rd, gt2_0[zctx0], gt2_1[zctx1], gt2_s[zctx_s]) == 0) return 2u;
        if (decode_tri(rd, gt3_0[zctx0], gt3_1[zctx1], gt3_s[zctx_s]) == 0) return 3u;

        if (decode_tri(rd, ge12_0[zctx0], ge12_1[zctx1], ge12_s[zctx_s]) == 0) {
            std::uint32_t tm = 1u;
            for (int i = 0; i < 3; ++i) {
                const int bit =
                    rd.decode_bit_prob(mix2_backoff(low_s[zctx_s][tm], low_0[zctx0][tm]));
                update_prob<4>(low_0[zctx0][tm], bit);
                update_prob<3>(low_s[zctx_s][tm], bit);
                tm = (tm << 1) | static_cast<std::uint32_t>(bit);
            }
            return (tm - 8u) + 4u;
        }

        int k = 0;
        while (decode_duo(rd, exp_s[zctx_s][std::min(k, 31)], exp_g[std::min(k, 31)]) == 1) {
            ++k;
            if (k > 30) {
                throw std::runtime_error("corrupted FZAC BWT zero-run");
            }
        }
        std::uint32_t u = 1u;
        const int bk = std::min(k, 31);
        for (int i = k - 1; i >= 0; --i) {
            const int bpos = std::min(k - 1 - i, 7);
            const int bit = decode_duo(rd, mant_s[zctx_s][bk][bpos], mant_g[bk][bpos]);
            u = (u << 1) | static_cast<std::uint32_t>(bit);
        }
        return u + 11u;
    }
};

struct BwtCtxState {
    std::uint8_t c0 = 0;
    std::uint32_t r_ctx = 0;
    std::uint32_t r_ctx14 = 0;
    std::uint32_t pair01 = 0;
    std::uint32_t pair23 = 0;
    std::uint32_t cr5 = 0;
};

/// Multi-context probability model for BWT + Sticky MTF-2 bit-trees.
struct BwtModel {
    std::array<std::uint16_t, 256> is_zero_c0{};
    std::array<std::uint16_t, 256> is_zero_rctx{};
    std::array<std::array<std::uint16_t, 256>, 256> is_zero_c0_r{};
    std::array<std::uint16_t, 16384> is_zero_hist{};
    std::array<std::array<std::uint16_t, 4>, 65536> is_zero_pair{};

    ZeroRunCoder zero_run;

    std::array<std::array<std::uint16_t, 8>, 256> exp_c0{};
    std::array<std::array<std::uint16_t, 8>, 256> exp_rctx{};
    std::array<std::array<std::array<std::uint16_t, 8>, 256>, 256> exp_c0_r{};
    std::array<std::array<std::uint16_t, 8>, 65536> exp_pair{};
    std::array<std::array<std::uint16_t, 8>, 16384> exp_hist{};

    std::array<std::array<std::uint16_t, 256>, 256> mant_c0{};
    std::array<std::array<std::uint16_t, 256>, 256> mant_rctx{};
    std::array<std::array<std::array<std::uint16_t, 256>, 32>, 256> mant_c0_r{};
    std::array<std::uint16_t, 65536> mant1_pair{};

    void init(bool large_bwt) {
        is_zero_c0.fill(kProbInit);
        is_zero_rctx.fill(kProbInit);
        zero_run.init();
        for (auto& r : exp_c0) r.fill(kProbInit);
        for (auto& r : exp_rctx) r.fill(kProbInit);
        for (auto& r : mant_c0) r.fill(kProbInit);
        for (auto& r : mant_rctx) r.fill(kProbInit);

        if (large_bwt) {
            for (auto& r : is_zero_c0_r) r.fill(kProbInit);
            is_zero_hist.fill(kProbInit);
            for (auto& r : is_zero_pair) r.fill(kProbInit);
            for (auto& m : exp_c0_r) {
                for (auto& r : m) r.fill(kProbInit);
            }
            for (auto& r : exp_pair) r.fill(kProbInit);
            for (auto& r : exp_hist) r.fill(kProbInit);
            for (auto& m : mant_c0_r) {
                for (auto& r : m) r.fill(kProbInit);
            }
            mant1_pair.fill(kProbInit);
        }
    }

    auto is_zero_prob(bool large_bwt, const BwtCtxState& cs,
                      std::uint32_t prev_zrun_b) const -> std::uint32_t {
        if (!large_bwt) {
            return mix2_backoff(is_zero_c0[cs.c0], is_zero_rctx[cs.r_ctx]);
        }
        return mix5_backoff(
            is_zero_rctx[cs.r_ctx],
            is_zero_c0[cs.c0],
            is_zero_hist[cs.r_ctx14],
            is_zero_c0_r[cs.c0][cs.r_ctx],
            is_zero_pair[cs.pair01][prev_zrun_b]);
    }

    void is_zero_update(bool large_bwt, const BwtCtxState& cs,
                        std::uint32_t prev_zrun_b, int bit) {
        update_prob<3>(is_zero_rctx[cs.r_ctx], bit);
        update_prob<4>(is_zero_c0[cs.c0], bit);
        if (large_bwt) {
            update_prob<4>(is_zero_hist[cs.r_ctx14], bit);
            update_prob<4>(is_zero_c0_r[cs.c0][cs.r_ctx], bit);
            update_prob<4>(is_zero_pair[cs.pair01][prev_zrun_b], bit);
        }
    }

    auto exp_prob(bool large_bwt, const BwtCtxState& cs,
                  std::uint32_t tm) const -> std::uint32_t {
        if (!large_bwt) {
            return mix2_backoff(exp_c0[cs.c0][tm], exp_rctx[cs.r_ctx][tm]);
        }
        return mix5_backoff(
            exp_rctx[cs.r_ctx][tm],
            exp_c0[cs.c0][tm],
            exp_hist[cs.r_ctx14][tm],
            exp_c0_r[cs.c0][cs.r_ctx][tm],
            exp_pair[cs.pair01][tm]);
    }

    void exp_update(bool large_bwt, const BwtCtxState& cs,
                    std::uint32_t tm, int bit) {
        update_prob<3>(exp_rctx[cs.r_ctx][tm], bit);
        update_prob<4>(exp_c0[cs.c0][tm], bit);
        if (large_bwt) {
            update_prob<4>(exp_hist[cs.r_ctx14][tm], bit);
            update_prob<4>(exp_c0_r[cs.c0][cs.r_ctx][tm], bit);
            update_prob<4>(exp_pair[cs.pair01][tm], bit);
        }
    }

    auto mant_prob(bool large_bwt, const BwtCtxState& cs,
                   std::uint32_t exp, std::uint32_t idx) const -> std::uint32_t {
        if (!large_bwt) {
            return mix2_backoff(mant_c0[cs.c0][idx], mant_rctx[cs.r_ctx][idx]);
        }
        if (exp == 1u) {
            return mix4_backoff(
                mant_rctx[cs.r_ctx][idx],
                mant_c0[cs.c0][idx],
                mant_c0_r[cs.c0][cs.cr5][idx],
                mant1_pair[cs.pair23]);
        }
        return mix3_backoff(
            mant_rctx[cs.r_ctx][idx],
            mant_c0[cs.c0][idx],
            mant_c0_r[cs.c0][cs.cr5][idx]);
    }

    void mant_update(bool large_bwt, const BwtCtxState& cs,
                     std::uint32_t exp, std::uint32_t idx, int bit) {
        update_prob<3>(mant_rctx[cs.r_ctx][idx], bit);
        update_prob<4>(mant_c0[cs.c0][idx], bit);
        if (large_bwt) {
            update_prob<4>(mant_c0_r[cs.c0][cs.cr5][idx], bit);
            if (exp == 1u) {
                update_prob<4>(mant1_pair[cs.pair23], bit);
            }
        }
    }
};

inline void mtf_sticky_move(std::array<std::uint8_t, 256>& mtf,
                            std::uint32_t rank, std::uint8_t val,
                            std::uint32_t prev_zrun_b) {
    if (rank > 1u && prev_zrun_b >= 2u) {
        std::memmove(mtf.data() + 2, mtf.data() + 1, rank - 1u);
        mtf[1] = val;
    } else {
        std::memmove(mtf.data() + 1, mtf.data(), rank);
        mtf[0] = val;
    }
}

void encode_bwt_mtf(RangeEncoder& rc, const std::uint8_t* L, std::uint32_t m) {
    if (m == 0u) return;
    const bool large_bwt = (m >= 262144u);
    auto bm = std::make_unique<BwtModel>();
    bm->init(large_bwt);

    std::array<std::uint8_t, 256> mtf{};
    for (int i = 0; i < 256; ++i) {
        mtf[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i);
    }

    std::uint32_t prev_rank_b = 0u;
    std::uint32_t prev2_rank_b = 0u;
    std::uint32_t prev3_rank_b = 0u;
    std::uint32_t prev4_rank_b = 0u;
    std::uint32_t prev_zrun_b = 0u;
    std::uint32_t last_zrun_b = 0u;
    bool after_zrun = false;

    std::uint32_t pos = 0u;
    while (pos < m) {
        BwtCtxState cs;
        cs.c0 = mtf[0];
        cs.r_ctx = (prev_zrun_b << 6) | (prev_rank_b << 3) | prev2_rank_b;
        cs.r_ctx14 = (prev_zrun_b << 12) | (prev_rank_b << 9) |
                     (prev2_rank_b << 6) | (prev3_rank_b << 3) | prev4_rank_b;
        cs.pair01 = (static_cast<std::uint32_t>(cs.c0) << 8) | mtf[1];
        cs.pair23 = (static_cast<std::uint32_t>(mtf[2]) << 8) | mtf[3];
        cs.cr5 = (prev_zrun_b << 3) | prev_rank_b;

        const bool is_z = (L[pos] == cs.c0);
        if (!after_zrun) {
            const int bit = is_z ? 1 : 0;
            rc.encode_bit_prob(bm->is_zero_prob(large_bwt, cs, prev_zrun_b), bit);
            bm->is_zero_update(large_bwt, cs, prev_zrun_b, bit);
        }

        if (is_z) {
            std::uint32_t zrun = 1u;
            while (pos + zrun < m && L[pos + zrun] == cs.c0) {
                ++zrun;
            }
            bm->zero_run.encode(rc, cs.c0, prev_rank_b, prev2_rank_b,
                                last_zrun_b, zrun);
            pos += zrun;
            prev_zrun_b = (zrun == 1u) ? 1u : ((zrun <= 3u) ? 2u : 3u);
            last_zrun_b = prev_zrun_b;
            after_zrun = true;
            continue;
        }

        const std::uint8_t target = L[pos];
        std::uint32_t rank = 1u;
        while (mtf[rank] != target) ++rank;
        mtf_sticky_move(mtf, rank, target, prev_zrun_b);

        const std::uint32_t exp = static_cast<std::uint32_t>(std::bit_width(rank)) - 1u;
        {
            std::uint32_t tree_m = 1u;
            for (int i = 2; i >= 0; --i) {
                const int bit = static_cast<int>((exp >> i) & 1u);
                rc.encode_bit_prob(bm->exp_prob(large_bwt, cs, tree_m), bit);
                bm->exp_update(large_bwt, cs, tree_m, bit);
                tree_m = (tree_m << 1) | static_cast<std::uint32_t>(bit);
            }
        }
        if (exp > 0u) {
            const std::uint32_t base = 1u << exp;
            const std::uint32_t mant = rank - base;
            std::uint32_t tm = 1u;
            for (int i = static_cast<int>(exp) - 1; i >= 0; --i) {
                const int bit = static_cast<int>((mant >> i) & 1u);
                const std::uint32_t idx = base + tm;
                rc.encode_bit_prob(bm->mant_prob(large_bwt, cs, exp, idx), bit);
                bm->mant_update(large_bwt, cs, exp, idx, bit);
                tm = (tm << 1) | static_cast<std::uint32_t>(bit);
            }
        }

        ++pos;
        prev4_rank_b = prev3_rank_b;
        prev3_rank_b = prev2_rank_b;
        prev2_rank_b = prev_rank_b;
        prev_rank_b = exp;
        prev_zrun_b = 0u;
        after_zrun = false;
    }
}

void decode_bwt_mtf(RangeDecoder& rd, std::uint8_t* L, std::uint32_t m) {
    if (m == 0u) return;
    const bool large_bwt = (m >= 262144u);
    auto bm = std::make_unique<BwtModel>();
    bm->init(large_bwt);

    std::array<std::uint8_t, 256> mtf{};
    for (int i = 0; i < 256; ++i) {
        mtf[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i);
    }

    std::uint32_t prev_rank_b = 0u;
    std::uint32_t prev2_rank_b = 0u;
    std::uint32_t prev3_rank_b = 0u;
    std::uint32_t prev4_rank_b = 0u;
    std::uint32_t prev_zrun_b = 0u;
    std::uint32_t last_zrun_b = 0u;
    bool after_zrun = false;

    std::uint32_t pos = 0u;
    while (pos < m) {
        BwtCtxState cs;
        cs.c0 = mtf[0];
        cs.r_ctx = (prev_zrun_b << 6) | (prev_rank_b << 3) | prev2_rank_b;
        cs.r_ctx14 = (prev_zrun_b << 12) | (prev_rank_b << 9) |
                     (prev2_rank_b << 6) | (prev3_rank_b << 3) | prev4_rank_b;
        cs.pair01 = (static_cast<std::uint32_t>(cs.c0) << 8) | mtf[1];
        cs.pair23 = (static_cast<std::uint32_t>(mtf[2]) << 8) | mtf[3];
        cs.cr5 = (prev_zrun_b << 3) | prev_rank_b;

        int is_z = 0;
        if (!after_zrun) {
            is_z = rd.decode_bit_prob(bm->is_zero_prob(large_bwt, cs, prev_zrun_b));
            bm->is_zero_update(large_bwt, cs, prev_zrun_b, is_z);
        }

        if (is_z == 1) {
            const std::uint32_t zrun = bm->zero_run.decode(
                rd, cs.c0, prev_rank_b, prev2_rank_b, last_zrun_b);
            if (zrun == 0u || zrun > m - pos) {
                throw std::runtime_error("FZAC corrupted BWT zero-run");
            }
            std::memset(L + pos, cs.c0, zrun);
            pos += zrun;
            prev_zrun_b = (zrun == 1u) ? 1u : ((zrun <= 3u) ? 2u : 3u);
            last_zrun_b = prev_zrun_b;
            after_zrun = true;
            continue;
        }

        std::uint32_t exp = 1u;
        for (int i = 0; i < 3; ++i) {
            const int bit = rd.decode_bit_prob(bm->exp_prob(large_bwt, cs, exp));
            bm->exp_update(large_bwt, cs, exp, bit);
            exp = (exp << 1) | static_cast<std::uint32_t>(bit);
        }
        exp -= 8u;

        std::uint32_t rank = 1u;
        if (exp > 0u) {
            const std::uint32_t base = 1u << exp;
            std::uint32_t tm = 1u;
            for (std::uint32_t i = 0; i < exp; ++i) {
                const std::uint32_t idx = base + tm;
                const int bit =
                    rd.decode_bit_prob(bm->mant_prob(large_bwt, cs, exp, idx));
                bm->mant_update(large_bwt, cs, exp, idx, bit);
                tm = (tm << 1) | static_cast<std::uint32_t>(bit);
            }
            rank = tm;
        }

        const std::uint8_t val = mtf[rank];
        mtf_sticky_move(mtf, rank, val, prev_zrun_b);
        L[pos] = val;

        ++pos;
        prev4_rank_b = prev3_rank_b;
        prev3_rank_b = prev2_rank_b;
        prev2_rank_b = prev_rank_b;
        prev_rank_b = exp;
        prev_zrun_b = 0u;
        after_zrun = false;
    }
}

auto compress_bwt_block(const std::uint8_t* raw, std::uint32_t n)
    -> std::vector<std::byte> {
    if (n == 0u) return {};

    std::vector<std::uint8_t> dict_syms;
    std::vector<std::uint8_t> dict_buf;
    const bool use_dict = dict_forward(raw, n, dict_syms, dict_buf);
    const std::uint8_t* stage1_ptr = use_dict ? dict_buf.data() : raw;
    const std::uint32_t stage1_len =
        use_dict ? static_cast<std::uint32_t>(dict_buf.size()) : n;

    std::uint8_t esc_byte = 0u;
    std::vector<std::uint8_t> lzp_buf;
    const bool use_lzp = lzp_forward(stage1_ptr, stage1_len, esc_byte, lzp_buf);
    if (use_lzp && use_dict) {
        std::vector<std::uint8_t> free_dict;
        dict_buf.swap(free_dict);
    }

    const std::uint8_t* bwt_in = use_lzp ? lzp_buf.data() : stage1_ptr;
    const std::uint32_t m =
        use_lzp ? static_cast<std::uint32_t>(lzp_buf.size()) : stage1_len;

    std::vector<std::uint8_t> L;
    std::uint32_t primary_idx = 0u;
    bwt_forward(bwt_in, m, L, primary_idx);
    {
        std::vector<std::uint8_t> free_lzp;
        lzp_buf.swap(free_lzp);
        std::vector<std::uint8_t> free_dict;
        dict_buf.swap(free_dict);
    }

    RangeEncoder rc;
    rc.reserve(m / 2u + 128u);
    write_header(rc.buffer(), kVersionSingle, /*filter_id=*/2u, /*lit_mode=*/0u, /*pb=*/0u, n);
    const std::uint8_t bwt_flags =
        static_cast<std::uint8_t>((use_lzp ? 1u : 0u) | (use_dict ? 2u : 0u));
    rc.buffer().push_back(static_cast<std::byte>(bwt_flags));
    if (use_dict) {
        rc.buffer().push_back(static_cast<std::byte>(dict_syms.size()));
        for (const std::uint8_t s : dict_syms) {
            rc.buffer().push_back(static_cast<std::byte>(s));
        }
        append_varint(rc.buffer(), stage1_len);
    }
    if (use_lzp) {
        rc.buffer().push_back(static_cast<std::byte>(esc_byte));
        append_varint(rc.buffer(), m);
    }
    append_varint(rc.buffer(), primary_idx);

    encode_bwt_mtf(rc, L.data(), m);
    rc.finish();

    if (rc.buffer().size() >= n) {
        return {};
    }
    return std::move(rc.buffer());
}

void decode_bwt_block(const std::byte* rd_begin, const std::byte* rd_end,
                      std::uint8_t* dst, std::uint32_t n) {
    if (n == 0u) return;
    const auto* in = reinterpret_cast<const std::uint8_t*>(rd_begin);
    const std::size_t avail = static_cast<std::size_t>(rd_end - rd_begin);
    std::size_t off = 0u;
    if (off >= avail) {
        throw std::runtime_error("truncated FZAC BWT header");
    }
    const std::uint8_t bwt_flags = in[off++];
    if ((bwt_flags & ~3u) != 0u) {
        throw std::runtime_error("unsupported FZAC BWT flags");
    }
    const bool use_lzp = (bwt_flags & 1u) != 0u;
    const bool use_dict = (bwt_flags & 2u) != 0u;

    std::vector<std::uint8_t> dict_syms;
    std::uint32_t stage1_len = n;
    if (use_dict) {
        if (off >= avail) {
            throw std::runtime_error("truncated FZAC BWT dict count");
        }
        const std::uint8_t dict_k = in[off++];
        if (dict_k == 0u || dict_k > kDictTokens.size()) {
            throw std::runtime_error("invalid FZAC BWT dict count");
        }
        if (avail - off < dict_k) {
            throw std::runtime_error("truncated FZAC BWT dict symbols");
        }
        dict_syms.resize(dict_k);
        std::array<bool, 256> seen{};
        for (std::size_t j = 0; j < dict_k; ++j) {
            const std::uint8_t sym = in[off++];
            if (seen[sym]) {
                throw std::runtime_error("duplicate FZAC BWT dict symbol");
            }
            seen[sym] = true;
            dict_syms[j] = sym;
        }
        const std::uint64_t s1_64 =
            read_varint(in, avail, off, "truncated FZAC BWT dict size");
        if (s1_64 == 0u || s1_64 > n) {
            throw std::runtime_error("invalid FZAC BWT dict size");
        }
        stage1_len = static_cast<std::uint32_t>(s1_64);
    }

    std::uint8_t esc_byte = 0u;
    std::uint32_t m = stage1_len;
    if (use_lzp) {
        if (off >= avail) {
            throw std::runtime_error("truncated FZAC BWT LZP escape");
        }
        esc_byte = in[off++];
        const std::uint64_t m64 =
            read_varint(in, avail, off, "truncated FZAC BWT LZP size");
        if (m64 == 0u || m64 > stage1_len) {
            throw std::runtime_error("invalid FZAC BWT LZP size");
        }
        m = static_cast<std::uint32_t>(m64);
    }

    const std::uint64_t p64 =
        read_varint(in, avail, off, "truncated FZAC BWT primary index");
    if (p64 >= m) {
        throw std::runtime_error("invalid FZAC BWT primary index");
    }
    const std::uint32_t primary_idx = static_cast<std::uint32_t>(p64);

    std::vector<std::uint8_t> L(m);
    RangeDecoder rd(rd_begin + off, rd_end);
    decode_bwt_mtf(rd, L.data(), m);

    if (!use_lzp && !use_dict) {
        bwt_inverse(L.data(), m, primary_idx, dst);
    } else if (use_lzp && !use_dict) {
        std::vector<std::uint8_t> lzp_buf(m);
        bwt_inverse(L.data(), m, primary_idx, lzp_buf.data());
        lzp_inverse(lzp_buf.data(), m, esc_byte, dst, n);
    } else if (!use_lzp && use_dict) {
        std::vector<std::uint8_t> dict_buf(stage1_len);
        bwt_inverse(L.data(), m, primary_idx, dict_buf.data());
        dict_inverse(dict_buf.data(), stage1_len, dict_syms.data(),
                     dict_syms.size(), dst, n);
    } else {
        std::vector<std::uint8_t> lzp_buf(m);
        bwt_inverse(L.data(), m, primary_idx, lzp_buf.data());
        {
            std::vector<std::uint8_t> free_L;
            L.swap(free_L);
        }
        std::vector<std::uint8_t> dict_buf(stage1_len);
        lzp_inverse(lzp_buf.data(), m, esc_byte, dict_buf.data(), stage1_len);
        {
            std::vector<std::uint8_t> free_lzp;
            lzp_buf.swap(free_lzp);
        }
        dict_inverse(dict_buf.data(), stage1_len, dict_syms.data(),
                     dict_syms.size(), dst, n);
    }
}

auto compress_single_block(const std::uint8_t* raw, std::uint32_t n, int level)
    -> std::vector<std::byte> {
    const bool use_bcj = should_enable_bcj(raw, n);
    std::vector<std::uint8_t> filtered;
    const std::uint8_t* src = raw;
    if (use_bcj) {
        filtered.assign(raw, raw + n);
        bcj_x86_forward(filtered.data(), n);
        src = filtered.data();
    }

    const std::uint8_t filter_id = use_bcj ? 1u : 0u;
    const std::uint8_t lit_mode = choose_lit_mode(n);
    const std::uint8_t pb = choose_pos_bits(src, n, use_bcj);

    if (!use_bcj && pb == 0u && n > 524288u && n <= (128u << 20)) {
        return compress_bwt_block(raw, n);
    }

    RangeEncoder rc;
    rc.reserve(n / 2u + 64u);
    write_header(rc.buffer(), kVersionSingle, filter_id, lit_mode, pb, n);

    ModelState model;
    model.init(lit_mode, pb, n);
    MatchFinder mf(src, n, level);

    if (level <= 6) {
        compress_fast_lazy(rc, model, mf, src, n, level);
    } else {
        compress_optimal(rc, model, mf, src, n, level);
    }
    rc.finish();

    std::vector<std::byte> lz_out;
    if (rc.buffer().size() < n) {
        lz_out = std::move(rc.buffer());
    }

    if (!use_bcj && pb == 0u && n >= 4096u && n <= 524288u && level >= 6) {
        auto bwt_out = compress_bwt_block(raw, n);
        if (!bwt_out.empty() && (lz_out.empty() || bwt_out.size() < lz_out.size())) {
            return bwt_out;
        }
    }
    return lz_out;
}

void decode_fzac_block(const std::byte* rd_begin, const std::byte* rd_end,
                       std::uint8_t filter_id, std::uint8_t lit_mode,
                       std::uint8_t pb, std::uint8_t* dst, std::uint32_t n) {
    if (n == 0u) return;
    if (filter_id == 2u) {
        decode_bwt_block(rd_begin, rd_end, dst, n);
        return;
    }

    ModelState m;
    m.init(lit_mode, pb, n);
    RangeDecoder rd(rd_begin, rd_end);

    std::uint8_t state = 0;
    std::array<std::uint32_t, 4> rep = {1u, 1u, 1u, 1u};

    // Position 0 is always a literal.
    dst[0] = decode_literal(rd, m, 0u, 0u, 0u, 0u, 0u, false, 0u);
    std::uint32_t pos = 1u;

    while (pos < n) {
        const std::uint32_t pos_state = pos & m.pos_mask;
        const std::uint8_t prev_b = dst[pos - 1u];
        const std::uint32_t match_ctx = match_ctx_of(pos, m.pos_mask, prev_b);
        const std::uint32_t rep_ctx = rep_ctx_of(pos, m.pos_mask, prev_b);
        if (rd.decode_dual(m.is_match[state][match_ctx]) == 0) {
            const std::uint8_t prev2_b = (pos > 1u) ? dst[pos - 2u] : 0u;
            const std::uint8_t prev3_b = (pos > 2u) ? dst[pos - 3u] : 0u;
            const std::uint8_t prev4_b = (pos > 3u) ? dst[pos - 4u] : 0u;
            const bool matched = (state >= 7u);
            const std::uint8_t mb = matched ? dst[pos - rep[0]] : 0u;
            dst[pos] = decode_literal(rd, m, pos, prev_b, prev2_b, prev3_b, prev4_b,
                                      matched, mb);
            ++pos;
            state = kNextStateLit[state];
            continue;
        }

        std::uint32_t len = 0;
        std::uint32_t dist = 0;
        if (rd.decode_dual(m.is_rep[state]) == 1) {
            if (rd.decode_dual(m.is_rep0[state]) == 0) {
                if (rd.decode_dual(m.is_rep0_long[state][rep_ctx]) == 0) {
                    if (rep[0] > pos) {
                        throw std::runtime_error("FZAC invalid shortrep distance");
                    }
                    dst[pos] = dst[pos - rep[0]];
                    ++pos;
                    state = state_after_shortrep(state);
                    continue;
                }
                dist = rep[0];
            } else {
                if (rd.decode_dual(m.is_rep1[state]) == 0) {
                    dist = rep[1];
                    rep[1] = rep[0];
                } else {
                    if (rd.decode_dual(m.is_rep2[state]) == 0) {
                        dist = rep[2];
                    } else {
                        dist = rep[3];
                        rep[3] = rep[2];
                    }
                    rep[2] = rep[1];
                    rep[1] = rep[0];
                }
                rep[0] = dist;
            }
            len = m.rep_len_coder.decode(rd, pos_state, kMinRepLen);
            state = state_after_rep(state);
        } else {
            len = m.match_len_coder.decode(rd, pos_state, kMinMatchLen);
            dist = decode_dist(rd, m, pos, len);
            rep[3] = rep[2];
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = dist;
            state = state_after_match(state);
        }

        if (dist == 0u || dist > pos || pos + len > n) {
            throw std::runtime_error("FZAC corrupted match bounds");
        }

        // Fast match copy: memset for RLE dist==1, memcpy when non-overlapping.
        std::uint8_t* write_p = dst + pos;
        const std::uint8_t* read_p = write_p - dist;
        if (dist == 1u) {
            std::memset(write_p, *read_p, len);
        } else if (dist >= len) {
            std::memcpy(write_p, read_p, len);
        } else {
            for (std::uint32_t k = 0; k < len; ++k) {
                write_p[k] = read_p[k];
            }
        }
        pos += len;
    }

    if (filter_id == 1u) {
        bcj_x86_inverse(dst, n);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public Compress & Decompress API
// ---------------------------------------------------------------------------
auto fzac_compress(std::span<const std::byte> data, int level)
    -> std::vector<std::byte> {
    if (data.empty()) return {};

    level = std::clamp(level, 1, 12);
    if (level >= 7) {
        return fzac_v3::fzac_v3_compress(data, level);
    }
    const auto* raw = reinterpret_cast<const std::uint8_t*>(data.data());
    const unsigned hw = detail::effective_threads();

    constexpr std::size_t kParallelThreshold = 4u << 20;  // 4 MiB
    if (hw <= 1u || data.size() <= kParallelThreshold) {
        if (data.size() > static_cast<std::size_t>(kNullPos - 1u)) {
            return {};
        }
        return compress_single_block(
            raw, static_cast<std::uint32_t>(data.size()), level);
    }

    std::size_t block_size = 4u << 20;
    if (data.size() >= (128u << 20)) {
        block_size = (level >= 9) ? (64u << 20)
                                  : ((level >= 7) ? (32u << 20) : (16u << 20));
    } else if (data.size() >= (32u << 20)) {
        block_size = (level >= 7) ? (16u << 20) : (8u << 20);
    }

    const std::size_t num_blocks = (data.size() + block_size - 1u) / block_size;
    std::vector<std::vector<std::byte>> block_out(num_blocks);
    std::vector<std::uint32_t> block_raw_len(num_blocks, 0u);
    std::atomic<std::size_t> next_blk{0};

    const unsigned max_nw = (block_size >= (64u << 20)) ? 8u : 12u;
    const unsigned nw = std::min<unsigned>(
        hw, std::min<unsigned>(static_cast<unsigned>(num_blocks), max_nw));
    std::vector<std::thread> workers;
    workers.reserve(nw);
    for (unsigned t = 0; t < nw; ++t) {
        workers.emplace_back([&]() {
            for (;;) {
                const std::size_t b = next_blk.fetch_add(1, std::memory_order_relaxed);
                if (b >= num_blocks) break;
                const std::size_t off = b * block_size;
                const std::uint32_t blen = static_cast<std::uint32_t>(
                    std::min<std::size_t>(block_size, data.size() - off));
                block_raw_len[b] = blen;
                block_out[b] = compress_single_block(raw + off, blen, level);
            }
        });
    }
    for (auto& w : workers) w.join();

    std::vector<std::byte> out;
    write_header(out, kVersionMultiBlock, 0u, 0u, 0u, data.size());
    append_varint(out, num_blocks);

    std::size_t payload_total = 0;
    for (std::size_t b = 0; b < num_blocks; ++b) {
        append_varint(out, block_raw_len[b]);
        if (block_out[b].empty()) {
            append_varint(out, 0u);
            payload_total += block_raw_len[b];
        } else {
            append_varint(out, block_out[b].size());
            payload_total += block_out[b].size();
        }
    }

    if (out.size() + payload_total >= data.size()) {
        return {};
    }

    out.reserve(out.size() + payload_total);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        if (block_out[b].empty()) {
            const std::size_t off = b * block_size;
            const std::byte* raw_b = data.data() + off;
            out.insert(out.end(), raw_b, raw_b + block_raw_len[b]);
        } else {
            out.insert(out.end(), block_out[b].begin(), block_out[b].end());
        }
    }
    return out;
}

auto fzac_decompress(std::span<const std::byte> data,
                     std::size_t expected_size) -> std::vector<std::byte> {
    if (data.size() < 4u) {
        throw std::runtime_error("FZAC stream too short");
    }
    const auto* in = reinterpret_cast<const std::uint8_t*>(data.data());
    if (in[0] != kMagic0 || in[1] != kMagic1) {
        throw std::runtime_error("invalid FZAC magic");
    }
    const std::uint8_t flags = in[2];
    const std::uint8_t version = flags & 0x03u;
    const std::uint8_t filter_id = (flags >> 2) & 0x03u;
    const std::uint8_t lit_mode = (flags >> 4) & 0x03u;
    const std::uint8_t pb = (flags >> 6) & 0x03u;

    std::size_t hdr_off = 3u;
    const std::uint64_t raw_size =
        read_varint(in, data.size(), hdr_off, "truncated FZAC header size");

    if (expected_size > 0 && raw_size != expected_size) {
        throw std::runtime_error("FZAC size mismatch");
    }

    if (version == kVersionSingle) {
        if (filter_id > 2u || pb > 2u) {
            throw std::runtime_error("unsupported FZAC filter or pb");
        }
        if (raw_size > static_cast<std::uint64_t>(kNullPos - 1u)) {
            throw std::runtime_error("FZAC output size too large");
        }
        const std::uint32_t n = static_cast<std::uint32_t>(raw_size);
        std::vector<std::byte> out(n);
        if (n == 0u) return out;

        decode_fzac_block(data.data() + hdr_off, data.data() + data.size(),
                          filter_id, lit_mode, pb,
                          reinterpret_cast<std::uint8_t*>(out.data()), n);
        return out;
    }

    if (version == kVersionCM) {
        if (raw_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("FZAC v3 output size exceeds addressable memory");
        }
        std::vector<std::byte> out(static_cast<std::size_t>(raw_size));
        if (raw_size == 0u) return out;
        fzac_v3::fzac_v3_decompress(
            data.data() + hdr_off, data.data() + data.size(), flags,
            reinterpret_cast<std::uint8_t*>(out.data()),
            static_cast<std::size_t>(raw_size));
        return out;
    }

    if (version != kVersionMultiBlock) {
        throw std::runtime_error("unsupported FZAC version");
    }
    if (filter_id != 0u || lit_mode != 0u || pb != 0u) {
        throw std::runtime_error("unsupported FZAC multi-block header flags");
    }

    const std::uint64_t num_blocks64 =
        read_varint(in, data.size(), hdr_off, "truncated FZAC block count");
    if (num_blocks64 < 2u || num_blocks64 > raw_size) {
        throw std::runtime_error("invalid FZAC multi-block count");
    }
    const std::size_t num_blocks = static_cast<std::size_t>(num_blocks64);

    struct BlockDesc {
        std::size_t in_off = 0;
        std::size_t comp_size = 0;
        std::size_t out_off = 0;
        std::uint32_t raw_len = 0;
        bool stored = false;
    };
    std::vector<BlockDesc> blocks(num_blocks);
    std::uint64_t sum_raw = 0;
    for (std::size_t b = 0; b < num_blocks; ++b) {
        const std::uint64_t br =
            read_varint(in, data.size(), hdr_off, "truncated FZAC block directory");
        const std::uint64_t bp =
            read_varint(in, data.size(), hdr_off, "truncated FZAC block directory");
        if (br == 0u || br > static_cast<std::uint64_t>(kNullPos - 1u)) {
            throw std::runtime_error("invalid FZAC block raw length");
        }
        if (bp > data.size()) {
            throw std::runtime_error("FZAC block payload exceeds stream size");
        }
        blocks[b].out_off = static_cast<std::size_t>(sum_raw);
        blocks[b].raw_len = static_cast<std::uint32_t>(br);
        blocks[b].stored = (bp == 0u);
        blocks[b].comp_size = static_cast<std::size_t>(bp);
        sum_raw += br;
    }
    if (sum_raw != raw_size) {
        throw std::runtime_error("FZAC multi-block size mismatch");
    }

    std::size_t in_cursor = hdr_off;
    for (auto& blk : blocks) {
        const std::size_t need = blk.stored ? blk.raw_len : blk.comp_size;
        if (need > data.size() - in_cursor) {
            throw std::runtime_error("truncated FZAC multi-block payload");
        }
        blk.in_off = in_cursor;
        in_cursor += need;
    }

    if (raw_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("FZAC output size exceeds addressable memory");
    }
    std::vector<std::byte> out(static_cast<std::size_t>(raw_size));
    auto* dst = reinterpret_cast<std::uint8_t*>(out.data());

    auto decode_one = [&](std::size_t b) {
        const BlockDesc& blk = blocks[b];
        if (blk.stored) {
            std::memcpy(dst + blk.out_off, data.data() + blk.in_off, blk.raw_len);
            return;
        }
        if (blk.comp_size < 4u) {
            throw std::runtime_error("FZAC sub-block too short");
        }
        const std::uint8_t* sub_in = in + blk.in_off;
        if (sub_in[0] != kMagic0 || sub_in[1] != kMagic1) {
            throw std::runtime_error("invalid FZAC sub-block magic");
        }
        const std::uint8_t sub_flags = sub_in[2];
        const std::uint8_t sub_ver = sub_flags & 0x03u;
        const std::uint8_t sub_filter = (sub_flags >> 2) & 0x03u;
        const std::uint8_t sub_lit = (sub_flags >> 4) & 0x03u;
        const std::uint8_t sub_pb = (sub_flags >> 6) & 0x03u;
        if (sub_ver != kVersionSingle || sub_filter > 2u || sub_pb > 2u) {
            throw std::runtime_error("unsupported FZAC sub-block header");
        }
        std::size_t sub_off = 3u;
        const std::uint64_t sub_raw =
            read_varint(sub_in, blk.comp_size, sub_off, "truncated FZAC sub-block header");
        if (sub_raw != blk.raw_len) {
            throw std::runtime_error("FZAC sub-block size mismatch");
        }
        decode_fzac_block(data.data() + blk.in_off + sub_off,
                          data.data() + blk.in_off + blk.comp_size,
                          sub_filter, sub_lit, sub_pb,
                          dst + blk.out_off, blk.raw_len);
    };

    const unsigned hw = detail::effective_threads();
    if (hw <= 1u) {
        for (std::size_t b = 0; b < num_blocks; ++b) {
            decode_one(b);
        }
    } else {
        const unsigned nw = std::min<unsigned>(
            hw, std::min<unsigned>(static_cast<unsigned>(num_blocks), 16u));
        std::atomic<std::size_t> next_blk{0};
        std::vector<std::exception_ptr> errors(nw, nullptr);
        std::vector<std::thread> workers;
        workers.reserve(nw);
        for (unsigned t = 0; t < nw; ++t) {
            workers.emplace_back([&, t]() {
                try {
                    for (;;) {
                        const std::size_t b = next_blk.fetch_add(1, std::memory_order_relaxed);
                        if (b >= num_blocks) break;
                        decode_one(b);
                    }
                } catch (...) {
                    errors[t] = std::current_exception();
                }
            });
        }
        for (auto& w : workers) w.join();
        for (const auto& e : errors) {
            if (e) std::rethrow_exception(e);
        }
    }

    return out;
}

}  // namespace fzip
