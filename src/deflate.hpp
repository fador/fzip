// fzip — DEFLATE encoder (RFC 1951) and matching inflate for round-trip.
//
// Hand-rolled implementation: LZ77 with a 3-byte hash + hash-chain match
// finder, a lazy matcher for better ratio, and dynamic Huffman trees
// (BTYPE=10). A fixed-Huffman (BTYPE=01) path is also provided for level 1
// or small inputs. Levels 1-9 trade match effort and lazy depth.
//
// The encoder emits a raw DEFLATE stream (no zlib/gzip wrapper), which is
// what the ZIP container stores as method-8 entry data. The final block is
// marked with BFINAL=1.
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

// Inflate a raw DEFLATE stream back to `expected` bytes. Used only by our
// own round-trip tests (the ZIP reader side is handled by 7za/zipfile).
auto deflate_decompress(std::span<const std::byte> data, std::size_t expected)
    -> std::vector<std::byte>;

}  // namespace fzip
