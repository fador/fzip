// fzip — Zstd codec (method 93) wrapper. Stage 4.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace fzip {

// Compress `data` with libzstd at `level` (1-22). If long_distance_log > 0,
// enables long-distance matching with that window log (up to 27 = 128 MiB).
// Returns compressed bytes; empty vector signals "fall back to Store".
auto zstd_compress(std::span<const std::byte> data, int level,
                   int long_distance_log = 0,
                   int target_cblock_size = 0) -> std::vector<std::byte>;

// Decompress a zstd stream of known `expected_size` bytes.
auto zstd_decompress(std::span<const std::byte> data, std::size_t expected_size)
    -> std::vector<std::byte>;

}  // namespace fzip
