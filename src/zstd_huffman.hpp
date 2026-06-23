// fzip — Huffman decoder for zstd literals (RFC 8878 §4.2).
// Used to decode literal bytes in compressed blocks.
#pragma once

#include <cstdint>
#include <vector>

namespace fzip::zstd {

constexpr int kHuffmanMaxSymbols = 256;
constexpr int kHuffmanMaxBits = 12;  // max code length in bits

// A Huffman decoding table entry.
struct HuffEntry {
    std::uint8_t symbol;
    std::uint8_t bits;     // number of bits consumed
};

// Huffman decoding table (direct lookup, no two-level).
struct HuffTable {
    int table_bits = 0;    // log2 of table size (typically 9)
    int table_size = 0;    // 1 << table_bits
    std::vector<HuffEntry> entries;
};

// Build a Huffman table from code lengths (per-symbol bit lengths).
// `lengths[s]` is the code length for symbol s (0 = not present).
// `max_symbol` is the highest symbol + 1.
auto build_huff_table(const int* lengths, int max_symbol) -> HuffTable;

// Decode one symbol from a forward bitstream using the Huffman table.
auto huff_decode_one(const HuffTable& table, const std::byte* data,
                     std::size_t size, std::size_t& bit_pos) -> std::uint8_t;

// Parse a Huffman weight table from a compressed block's literal section.
// The weights are themselves FSE-encoded. Returns the code lengths.
// `data` points to the start of the weight stream.
// `max_symbol` is the number of symbols (from the literals block header).
auto parse_huffman_weights(const std::byte* data, std::size_t size,
                           int max_symbol) -> std::vector<int>;

// Decode a single stream of Huffman-coded literals.
auto decode_huffman_stream(const HuffTable& table,
                           const std::byte* data, std::size_t size,
                           int num_literals) -> std::vector<std::byte>;

}  // namespace fzip::zstd
