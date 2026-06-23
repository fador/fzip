// fzip — FSE (Finite State Entropy) decoder for zstd (RFC 8878 §4.1).
// FSE is a variant of tANS used for all sequence symbols in zstd.
#pragma once

#include <cstdint>
#include <vector>

namespace fzip::zstd {

// Maximum number of symbols in an FSE table.
constexpr int kFseMaxSymbols = 256;
// Maximum accuracy log (table size = 1 << accuracyLog).
constexpr int kFseMaxAccuracyLog = 9;
// Maximum table size.
constexpr int kFseMaxTableSize = 1 << kFseMaxAccuracyLog;

// One entry in the FSE decoding table.
struct FseEntry {
    std::uint16_t new_state;  // next state after reading bits
    std::uint8_t  symbol;     // symbol decoded from this state
    std::uint8_t  bits;       // number of extra bits to read
};

// An FSE decoding table.
struct FseTable {
    int accuracy_log = 0;
    int table_size = 0;  // 1 << accuracy_log
    FseEntry entries[kFseMaxTableSize]{};
};

// Build an FSE decoding table from a normalized distribution.
// `accuracy_log` is the log2 of the table size (typically 5-9).
// `norm_counts[s]` is the normalized count for symbol s (sum = 1 << accuracy_log).
// Symbols with count 0 are not present. Symbols with count -1 get a "baseline"
// count of 1 (special zstd convention for probability 1/accuracyLog).
auto build_fse_table(int accuracy_log, const int* norm_counts, int max_symbol)
    -> FseTable;

// FSE reverse bitstream reader. The FSE bitstream is written forward (LSB
// but with state transitions), then read backward. The last byte written
// contains a 1-bit sentinel followed by zero-padding.
class FseBitReader {
  public:
    // `data` points to the END of the FSE bitstream (i.e. the last byte
    // written). `size` is the number of bytes in the bitstream.
    explicit FseBitReader(const std::byte* data, std::size_t size);

    // Read `n` bits (1-24) from the reverse bitstream.
    auto read_bits(int n) -> std::uint32_t;

    // Get the current FSE state (initial state = first `accuracy_log` bits).
    auto get_state(int accuracy_log) -> std::uint32_t;

    // Check if we've consumed all bits.
    auto empty() const -> bool;

  private:
    const std::byte* data_;  // points to end of stream
    std::size_t size_;       // total bytes
    std::size_t pos_;        // current bit position (counting from end backward)
};

// Decode one FSE symbol: read from state table, update state via bitstream.
auto fse_decode_one(const FseTable& table, FseBitReader& reader,
                    std::uint32_t& state) -> std::uint8_t;

// Parse an FSE table description from a bitstream (at the start of a
// compressed block's entropy section). Returns the table and advances
// the bitstream past the description.
// `max_symbol` is the maximum valid symbol + 1.
// `accuracy_log` is read from the first few bits.
auto parse_fse_table_description(const std::byte* data, std::size_t size,
                                 int max_symbol) -> FseTable;

// Predefined FSE tables for zstd sequences (RFC 8878 §4.1.1).
// These are used when the block header indicates "predefined" mode.
auto predefined_litlen_table() -> const FseTable&;
auto predefined_offset_table() -> const FseTable&;
auto predefined_matchlen_table() -> const FseTable&;

}  // namespace fzip::zstd
