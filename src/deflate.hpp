// fzip — DEFLATE encoder (method 8). Stage 3.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace fzip {

// Compress `data` into a raw DEFLATE stream (RFC 1951) using `level` (1-9).
// Returns the compressed bytes; caller writes them as method-8 entry data.
// If compression does not shrink the input, returns an empty vector and the
// caller should fall back to the Store method.
auto deflate_compress(std::span<const std::byte> data, int level)
    -> std::vector<std::byte>;

// Optional: inflate for our own round-trip tests. Not used by the writer.
auto deflate_decompress(std::span<const std::byte> data, std::size_t expected)
    -> std::vector<std::byte>;

}  // namespace fzip
