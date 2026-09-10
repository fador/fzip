// fzip — zstd Huffman codec for literals (RFC 8878 §4.2.1).
//
// Implements the 4-stream "Compressed_Literals_Block" layout with a directly
// encoded weight table. The bit conventions match the reference HUF codec:
// each stream is written forward and read backward, and every code is
// recovered MSB-first.
#pragma once

#include <cstdint>
#include <vector>

namespace fzip::zstd {

struct HufTable {
    int table_log = 0;                 // maximum code length
    std::vector<int> lengths;          // per-symbol code length (0 = absent)
    std::vector<std::uint32_t> codes;  // canonical code per symbol (MSB-first)
};

// Build a length-limited (<= max_bits) Huffman table from symbol frequencies.
auto huf_build(const std::vector<std::uint32_t>& freqs, int max_bits) -> HufTable;

// Compress literal bytes into the body of a Compressed_Literals_Block:
// weight table + jump table + 4 Huffman streams. Returns false if Huffman is
// not applicable (fewer than 2 symbols, or a symbol > 128 is present so the
// weights cannot be written directly). On success `out` holds the body whose
// size is the Compressed_Size (excluding the literals-section header).
auto huf_compress_literals(const std::uint8_t* literals, int num_literals,
                           std::vector<std::byte>& out) -> bool;

// Parse a weight table. Fills `t` and sets `consumed`. Returns false for
// unsupported formats (e.g. FSE-compressed weights).
auto huf_read_weights(const std::byte* data, std::size_t size, HufTable& t,
                      std::size_t& consumed) -> bool;

// Decode `num_literals` bytes from a 4-stream Huffman body that begins at
// `streams` (= the byte after the weight table), with `num_streams` streams
// per `size_format`. `body_size` is the remaining Compressed_Size.
auto huf_decode_streams(const HufTable& t, const std::byte* streams,
                        std::size_t body_size, int num_literals,
                        int size_format) -> std::vector<std::byte>;

}  // namespace fzip::zstd
