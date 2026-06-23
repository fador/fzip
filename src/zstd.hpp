// fzip — Custom zstd codec public API (RFC 8878).
// Replaces the vendored libzstd submodule.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fzip::zstd {

// Compress `data` into a valid zstd frame at `level` (1-22).
// Returns compressed bytes; empty vector signals "fall back to Store".
auto compress(std::span<const std::byte> data, int level,
              int long_distance_log = 0) -> std::vector<std::byte>;

// Decompress a zstd frame. `expected_size` is a hint (0 = unknown).
// Throws on format error or CRC mismatch.
auto decompress(std::span<const std::byte> data, std::size_t expected_size = 0)
    -> std::vector<std::byte>;

}  // namespace fzip::zstd
