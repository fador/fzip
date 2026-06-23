// fzip — CRC-32 implementation (IEEE polynomial 0xEDB88320, table-driven).
#include "crc32.hpp"

namespace fzip {

namespace {

// 256-entry table built lazily on first use. constexpr-compatible.
struct CrcTable {
    std::uint32_t entries[256]{};
    consteval CrcTable() : entries{} {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            entries[i] = c;
        }
    }
};

constexpr CrcTable kTable{};

}  // namespace

auto crc32_update(std::uint32_t crc, std::span<const std::byte> data)
    -> std::uint32_t {
    std::uint32_t c = crc ^ 0xFFFFFFFFu;
    for (auto b : data) {
        c = kTable.entries[(c ^ static_cast<std::uint8_t>(b)) & 0xFFu] ^
            (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

auto crc32(std::span<const std::byte> data) -> std::uint32_t {
    return crc32_update(0u, data);
}

}  // namespace fzip
