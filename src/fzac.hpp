// fzip — FZAC (fzip Binary Arithmetic Codec) encoder and decoder.
//
// Custom state-of-the-art lossless compressor/decompressor combining:
//   - Carryless 32-bit/64-bit binary arithmetic coding (range coding)
//   - Dual-rate / dual-context probability mixing
//   - 12-state LZ77 + 4-repeat-offset (rep0..rep3 + ShortRep) packet model
//   - Matched-literal bit-trees & adaptive Order-1 / Order-2 literal contexts
//   - Reversible x86 BCJ (E8/E9) pre-filter with automatic duplicate-target guard
//   - Hash2 + Hash3 + BT4 match finder with multi-pass forward DP optimal parser
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace fzip {

// Compress `data` using FZAC at `level` (1..12).
// Returns compressed bytes; returns an empty vector if compression does not
// shrink `data` (signaling callers to fall back to Store).
auto fzac_compress(std::span<const std::byte> data, int level = 9)
    -> std::vector<std::byte>;

// Decompress an FZAC stream back to its original bytes.
// If `expected_size > 0`, validates that the stream header matches it.
// Throws `std::runtime_error` on malformed or truncated input.
auto fzac_decompress(std::span<const std::byte> data,
                     std::size_t expected_size = 0) -> std::vector<std::byte>;

}  // namespace fzip
