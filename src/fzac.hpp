// fzip — FZAC (fzip Binary Arithmetic Codec) encoder and decoder.
//
// Custom state-of-the-art lossless compressor/decompressor:
//   - Levels 1..6: Fast LZ77 + repeat-offset + BWT/LZP binary arithmetic codec
//   - Levels 7..12 (FZAC v3): Deep Context Mixing (CM) engine combining:
//       * Dynamic Universal Word & Structural Token Transform (WRT) with implicit spaces
//       * Bijective x86/x64 E8/E9 + 0F 8x Jcc executable pre-filter
//       * Exact Direct O0/O1/O1.5/O2 models with bpos-indexed StateMaps
//       * Split Low/High 3-way 64-byte cache-line ContextMaps (O3..O15, Sparse, Word, XML, Record, Indirect, Exe)
//       * 4-way associative long-range Dual MatchModel with 1-byte edit recovery
//       * Variable-order byte-level PPMd (up to Order-16) with exclusion & SEE
//       * Multi-layer online byte-level coupled-gate LSTM (levels 10..12, bit-exact AVX2)
//       * 2-stage multi-context AVX2 Logistic Mixer + 4-stage count-adaptive APM/SSE
//       * 16-bit precision carryless range coding & parallel large-block scaling (<= 10 GB RAM)
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
