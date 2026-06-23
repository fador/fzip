// fzip — xxHash-64 implementation (used by zstd for content checksums).
// Minimal implementation: just the 64-bit hash, no streaming API.
#pragma once

#include <cstdint>
#include <span>

namespace fzip {

// Compute xxHash-64 of `data` with `seed` (default 0).
auto xxhash64(std::span<const std::byte> data, std::uint64_t seed = 0)
    -> std::uint64_t;

}  // namespace fzip
