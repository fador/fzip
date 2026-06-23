// fzip — CRC-32 (IEEE 802.3, polynomial 0xEDB88320, reflected).
// Used by the ZIP format for the CRC-32 of uncompressed entry data.
#pragma once

#include <cstdint>
#include <span>

namespace fzip {

// Continue a CRC-32 computation. `crc` starts at 0 on the first call; XOR
// the final result with 0xFFFFFFFF to obtain the stored ZIP CRC-32 value.
auto crc32_update(std::uint32_t crc, std::span<const std::byte> data)
    -> std::uint32_t;

// One-shot CRC-32 over a whole buffer, returning the ZIP-format value
// (i.e. already inverted to the canonical representation).
auto crc32(std::span<const std::byte> data) -> std::uint32_t;

}  // namespace fzip
