// fzip — xxHash-64 implementation.
// Based on the xxHash specification by Yann Collet.
#include "xxhash.hpp"

namespace fzip {

namespace {

constexpr std::uint64_t kPrime64_1 = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t kPrime64_2 = 0xC2B2AE3D27D4EB4FULL;
constexpr std::uint64_t kPrime64_3 = 0x165667B19E3779F9ULL;
constexpr std::uint64_t kPrime64_4 = 0x85EBCA77C2B2AE63ULL;
constexpr std::uint64_t kPrime64_5 = 0x27D4EB2F165667C5ULL;

inline auto rotl64(std::uint64_t x, int r) -> std::uint64_t {
    return (x << r) | (x >> (64 - r));
}

inline auto read_u64(const std::byte* p) -> std::uint64_t {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

inline auto read_u32(const std::byte* p) -> std::uint32_t {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline auto round64(std::uint64_t acc, std::uint64_t input) -> std::uint64_t {
    acc += input * kPrime64_2;
    acc = rotl64(acc, 31);
    acc *= kPrime64_1;
    return acc;
}

inline auto merge_round64(std::uint64_t acc, std::uint64_t val) -> std::uint64_t {
    val = round64(0, val);
    acc ^= val;
    acc = acc * kPrime64_1 + kPrime64_4;
    return acc;
}

inline auto avalanche(std::uint64_t h) -> std::uint64_t {
    h ^= h >> 33;
    h *= kPrime64_2;
    h ^= h >> 29;
    h *= kPrime64_3;
    h ^= h >> 32;
    return h;
}

}  // namespace

auto xxhash64(std::span<const std::byte> data, std::uint64_t seed)
    -> std::uint64_t {
    const auto* p = data.data();
    std::size_t len = data.size();
    std::uint64_t h;

    if (len >= 32) {
        const auto* end = p + len;
        std::uint64_t v1 = seed + kPrime64_1 + kPrime64_2;
        std::uint64_t v2 = seed + kPrime64_2;
        std::uint64_t v3 = seed + 0;
        std::uint64_t v4 = seed - kPrime64_1;

        while (p + 32 <= end) {
            v1 = round64(v1, read_u64(p)); p += 8;
            v2 = round64(v2, read_u64(p)); p += 8;
            v3 = round64(v3, read_u64(p)); p += 8;
            v4 = round64(v4, read_u64(p)); p += 8;
        }

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        h = merge_round64(h, v1);
        h = merge_round64(h, v2);
        h = merge_round64(h, v3);
        h = merge_round64(h, v4);
    } else {
        h = seed + kPrime64_5;
    }

    h += static_cast<std::uint64_t>(len);

    // Process remaining bytes in 8-byte, 4-byte, and 1-byte chunks.
    while (p + 8 <= data.data() + len) {
        std::uint64_t k1 = round64(0, read_u64(p));
        h ^= k1;
        h = rotl64(h, 27) * kPrime64_1 + kPrime64_4;
        p += 8;
    }
    if (p + 4 <= data.data() + len) {
        h ^= static_cast<std::uint64_t>(read_u32(p)) * kPrime64_1;
        h = rotl64(h, 23) * kPrime64_2 + kPrime64_3;
        p += 4;
    }
    while (p < data.data() + len) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(*p)) * kPrime64_5;
        h = rotl64(h, 11) * kPrime64_1;
        ++p;
    }

    return avalanche(h);
}

}  // namespace fzip
