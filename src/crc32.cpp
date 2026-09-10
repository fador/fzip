// fzip — CRC-32 implementation (IEEE polynomial 0xEDB88320, slicing-by-8).
#include "crc32.hpp"

namespace fzip {

namespace {

// Eight 256-entry tables built at compile time. Table 0 is the classic
// byte-at-a-time table; table k is derived from table k-1 by a right shift
// with byte reduction. Slicing-by-8 consumes 8 input bytes per iteration,
// which is roughly 4-8x faster than the byte-at-a-time loop on x86-64.
struct CrcSlicingTable {
    std::uint32_t t[8][256]{};
    consteval CrcSlicingTable() : t{} {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[0][i] = c;
        }
        for (int k = 1; k < 8; ++k) {
            for (std::uint32_t i = 0; i < 256; ++i) {
                t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xFFu];
            }
        }
    }
};

constexpr CrcSlicingTable kTable{};

}  // namespace

auto crc32_update(std::uint32_t crc, std::span<const std::byte> data)
    -> std::uint32_t {
    std::uint32_t c = crc ^ 0xFFFFFFFFu;
    const auto* p = reinterpret_cast<const std::uint8_t*>(data.data());
    std::size_t n = data.size();

    while (n >= 8) {
        // Load the first 8 bytes as two little-endian 32-bit words. Doing the
        // assembly by hand keeps this correct on big-endian hosts too.
        std::uint32_t lo = static_cast<std::uint32_t>(p[0]) |
                           (static_cast<std::uint32_t>(p[1]) << 8) |
                           (static_cast<std::uint32_t>(p[2]) << 16) |
                           (static_cast<std::uint32_t>(p[3]) << 24);
        std::uint32_t hi = static_cast<std::uint32_t>(p[4]) |
                           (static_cast<std::uint32_t>(p[5]) << 8) |
                           (static_cast<std::uint32_t>(p[6]) << 16) |
                           (static_cast<std::uint32_t>(p[7]) << 24);
        c ^= lo;
        c = kTable.t[7][c & 0xFFu] ^
            kTable.t[6][(c >> 8) & 0xFFu] ^
            kTable.t[5][(c >> 16) & 0xFFu] ^
            kTable.t[4][(c >> 24) & 0xFFu] ^
            kTable.t[3][hi & 0xFFu] ^
            kTable.t[2][(hi >> 8) & 0xFFu] ^
            kTable.t[1][(hi >> 16) & 0xFFu] ^
            kTable.t[0][(hi >> 24) & 0xFFu];
        p += 8;
        n -= 8;
    }

    while (n-- > 0) {
        c = kTable.t[0][(c ^ *p++) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

auto crc32(std::span<const std::byte> data) -> std::uint32_t {
    return crc32_update(0u, data);
}

}  // namespace fzip
