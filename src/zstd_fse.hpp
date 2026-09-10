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

    // Read `n` bits without advancing. Bits past the start of the stream read
    // as 0 (used by the Huffman decoder, whose table lookup only depends on
    // the leading bits of the peeked value).
    auto peek_bits(int n) -> std::uint32_t;

    // Like read_bits, but bits past the start of the stream read as 0 instead
    // of throwing (used by the FSE weight decoder).
    auto read_bits_padded(int n) -> std::uint32_t;

    // Skip `n` bits without reading.
    void skip_bits(int n) { pos_ += static_cast<std::size_t>(n); }

    // Current bit position (counted from the end of the stream).
    auto bit_pos() const -> std::size_t { return pos_; }

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
// These are used when the block header says "predefined" mode.
auto predefined_litlen_table() -> const FseTable&;
auto predefined_offset_table() -> const FseTable&;
auto predefined_matchlen_table() -> const FseTable&;

// Predefined normalized counts (for building encode tables that match
// the predefined decode tables the decoder uses).
auto predefined_litlen_norm() -> const int*;
auto predefined_offset_norm() -> const int*;
auto predefined_matchlen_norm() -> const int*;

// --- Encoder API ---

// Forward-direction bitstream writer (LSB-first).
class FseBitWriter {
  public:
    void put_bits(std::uint32_t value, int n);
    void put_bit(bool bit);
    void align_to_byte();
    void put_byte(std::uint8_t byte);
    auto data() const -> const std::vector<std::byte>&;
    auto size() const -> std::size_t;
    auto bit_count() const -> std::size_t;
    void clear();

  private:
    std::vector<std::byte> out_;
    std::uint32_t acc_ = 0;
    int acc_bits_ = 0;
};

// One entry in the FSE encoding table.
struct FseEncodeEntry {
    std::uint16_t baseline;  // base next-state
    std::uint8_t  bits;      // number of extra bits
    std::uint8_t  symbol;    // symbol at this position (for encoding lookup)
};

// FSE encoding table (forward direction). Built from the same spread as
// the decode table so positions match.
struct FseEncodeTable {
    int accuracy_log = 0;
    int table_size = 0;
    std::vector<FseEncodeEntry> entries;
    // symbol_start[s] = first table position for symbol s (from spread)
    std::vector<int> symbol_start;
    // symbol_count[s] = number of table positions for symbol s
    std::vector<int> symbol_count;
};

// Normalize symbol frequencies to a distribution with sum = 1 << accuracy_log.
auto fse_normalize(const int* freqs, int num_symbols, int accuracy_log)
    -> std::vector<int>;

// Build an FSE encoding table from normalized counts.
// Uses the same spread algorithm as build_fse_table so positions match.
auto build_fse_encode_table(int accuracy_log, const int* norm_counts,
                            int max_symbol) -> FseEncodeTable;

// Encode one symbol using the FSE encoding table.
// Updates the bitstream writer and the encoder state.
void fse_encode_one(FseBitWriter& writer, const FseEncodeTable& table,
                    std::uint32_t& state, std::uint8_t symbol);

// Flush the FSE encoder state (write the final state value).
void fse_flush_state(FseBitWriter& writer, const FseEncodeTable& table,
                     std::uint32_t state);

// Write an FSE table description to the bitstream (for inline table mode).
void write_fse_table_description(FseBitWriter& writer, int accuracy_log,
                                 const int* norm_counts, int max_symbol);

// --- Spec-correct FSE encoder (zstd v1.5.7 algorithm) ---
//
// `FseCTable` mirrors the reference FSE_CTable: a per-symbol transform plus a
// state-transition table. It is built from a normalized distribution that may
// contain -1 entries ("less than 1" probability), exactly like the predefined
// zstd distributions.
struct FseCTable {
    int table_log = 0;
    int table_size = 0;
    std::vector<std::uint16_t> state_table;      // size = table_size
    std::vector<std::int32_t> delta_find_state;  // per symbol
    std::vector<std::uint32_t> delta_nb_bits;    // per symbol
};

// Build an encode table from a normalized distribution (sum = 1<<table_log,
// values in {-1,0,1,...}). `norm` must have (max_symbol+1) entries.
auto build_fse_ctable(int table_log, const std::int16_t* norm, int max_symbol)
    -> FseCTable;

// FSE_initCState2: initialize a state with `symbol` (the first symbol the
// encoder emits, which is the last one the decoder reads).
auto fse_init_cstate2(const FseCTable& ct, int symbol) -> std::uint32_t;

// FSE_encodeSymbol: emit `symbol` from `value` and advance the state.
void fse_encode_symbol(FseBitWriter& w, const FseCTable& ct,
                       std::uint32_t& value, int symbol);

// FSE_flushCState: emit the final state (the decoder's initial state).
void fse_flush_cstate(FseBitWriter& w, const FseCTable& ct,
                      std::uint32_t value);

// --- Shared FSE table helpers (NCount + normalization + decode table) ---

// Normalize frequencies to a distribution summing to 1<<table_log. Symbols
// with zero frequency get zero; present symbols get at least 1. No -1
// entries (useLowProbCount = 0, as zstd uses for Huffman weights).
void fse_normalize(const unsigned* freqs, int total, int max_symbol,
                   int table_log, std::vector<int>& norm);

// Write/read an FSE table description (FSE_writeNCount / FSE_readNCount).
void fse_write_ncount(std::vector<std::byte>& out, const int* norm,
                      int max_symbol, int table_log);
auto fse_read_ncount(const std::byte* data, std::size_t size, int& table_log,
                     std::vector<int>& norm, int& max_symbol,
                     std::size_t& consumed) -> bool;

// A decoded FSE table: state -> {symbol, nb_bits, next_state}.
struct FseDecodeTable {
    int table_log = 0;
    std::vector<std::uint8_t> symbol;
    std::vector<std::uint8_t> nb_bits;
    std::vector<std::uint16_t> next_state;
};

// Build a decode table from normalized counts (FSE_buildDTable).
void build_fse_dtable(int table_log, const int* norm, int max_symbol,
                      FseDecodeTable& dt);

}  // namespace fzip::zstd
