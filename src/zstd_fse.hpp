// fzip — FSE (Finite State Entropy) decoder for zstd (RFC 8878 §4.1).
// FSE is a variant of tANS used for all sequence symbols in zstd.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "zstd_internal.hpp"

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
    // `data` points to the start of the FSE bitstream; `size` is the number of
    // bytes in the bitstream.
    explicit FseBitReader(const std::byte* data, std::size_t size)
        : data_(reinterpret_cast<const std::uint8_t*>(data)),
          total_bits_(size * 8),
          pos_(0) {
        if (size == 0) {
            bits_consumed_ = 64;
            return;
        }
        std::size_t idx = size;
        while (idx > 0) {
            --idx;
            const std::uint8_t byte = data_[idx];
            if (byte != 0) {
                const int lead = std::countl_zero(byte);
                pos_ = (size - 1 - idx) * 8 + static_cast<std::size_t>(lead) + 1;
                ptr_ = idx + 1;
                load_container();
                bits_consumed_ = lead + 1;
                return;
            }
        }
        ptr_ = size;
        load_container();
        bits_consumed_ = 0;
    }

    // Read `n` bits (0-28) from the reverse bitstream.
    auto read_bits(int n) -> std::uint32_t {
        if (n <= 0) return 0;
        if (pos_ + static_cast<std::size_t>(n) > total_bits_) {
            throw ZstdError("fse bitreader: underflow");
        }
        const std::uint32_t v = peek_bits(n);
        pos_ += static_cast<std::size_t>(n);
        bits_consumed_ += n;
        return v;
    }

    // Read `n` bits without advancing. Bits past the start of the stream read
    // as 0.
    auto peek_bits(int n) -> std::uint32_t {
        if (n <= 0) return 0;
        if (bits_consumed_ + n > 64 && ptr_ > 8) reload();
        if (bits_consumed_ >= 64) return 0;
        return static_cast<std::uint32_t>(
            (bit_container_ << bits_consumed_) >> (64 - n));
    }

    // Like read_bits, but bits past the start of the stream read as 0 instead
    // of throwing (used by the FSE weight decoder).
    auto read_bits_padded(int n) -> std::uint32_t {
        if (n <= 0) return 0;
        const std::uint32_t v = peek_bits(n);
        pos_ += static_cast<std::size_t>(n);
        bits_consumed_ += n;
        return v;
    }

    // Skip `n` bits without reading.
    void skip_bits(int n) {
        pos_ += static_cast<std::size_t>(n);
        bits_consumed_ += n;
        if (bits_consumed_ >= 32 && ptr_ > 8) reload();
    }

    // Current bit position (counted from the end of the stream).
    auto bit_pos() const -> std::size_t { return pos_; }

    // Get the current FSE state (initial state = first `accuracy_log` bits).
    auto get_state(int accuracy_log) -> std::uint32_t {
        return read_bits(accuracy_log);
    }

    // Check if we've consumed all bits.
    auto empty() const -> bool { return pos_ >= total_bits_; }

  private:
    void load_container() {
        if (ptr_ >= 8) {
            if constexpr (std::endian::native == std::endian::little) {
                std::memcpy(&bit_container_, data_ + ptr_ - 8, 8);
            } else {
                bit_container_ = 0;
                for (int i = 0; i < 8; ++i) {
                    bit_container_ |=
                        static_cast<std::uint64_t>(data_[ptr_ - 8 + i]) << (i * 8);
                }
            }
        } else {
            bit_container_ = 0;
            for (std::size_t i = 0; i < ptr_; ++i) {
                bit_container_ |=
                    static_cast<std::uint64_t>(data_[i]) << ((8 - ptr_ + i) * 8);
            }
        }
    }

    void reload() {
        if (ptr_ <= 8 || bits_consumed_ < 8) return;
        const std::size_t bytes = static_cast<std::size_t>(bits_consumed_ >> 3);
        const std::size_t step = std::min(bytes, ptr_ - 8);
        ptr_ -= step;
        bits_consumed_ -= static_cast<int>(step << 3);
        load_container();
    }

    const std::uint8_t* data_;
    std::size_t total_bits_;
    std::size_t pos_;
    std::size_t ptr_ = 0;
    std::uint64_t bit_container_ = 0;
    int bits_consumed_ = 0;
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

// Forward-direction bitstream writer (LSB-first, 64-bit accumulator).
class FseBitWriter {
  public:
    void reserve(std::size_t bytes) { out_.reserve(bytes); }

    void put_bits(std::uint32_t value, int n) {
        acc_ |= (static_cast<std::uint64_t>(value) & ((1ULL << n) - 1ULL)) << acc_bits_;
        acc_bits_ += n;
        if (acc_bits_ >= 32) {
            const std::size_t pos = out_.size();
            out_.resize(pos + 4);
            const std::uint32_t lo = static_cast<std::uint32_t>(acc_ & 0xFFFFFFFFULL);
            if constexpr (std::endian::native == std::endian::little) {
                std::memcpy(out_.data() + pos, &lo, 4);
            } else {
                out_[pos]     = static_cast<std::byte>(lo & 0xFFu);
                out_[pos + 1] = static_cast<std::byte>((lo >> 8) & 0xFFu);
                out_[pos + 2] = static_cast<std::byte>((lo >> 16) & 0xFFu);
                out_[pos + 3] = static_cast<std::byte>((lo >> 24) & 0xFFu);
            }
            acc_ >>= 32;
            acc_bits_ -= 32;
        }
    }

    void put_bit(bool bit) { put_bits(bit ? 1u : 0u, 1); }

    void align_to_byte() {
        while (acc_bits_ > 0) {
            out_.push_back(static_cast<std::byte>(acc_ & 0xFFu));
            acc_ >>= 8;
            acc_bits_ -= 8;
        }
        acc_ = 0;
        acc_bits_ = 0;
    }

    void put_byte(std::uint8_t byte) { out_.push_back(static_cast<std::byte>(byte)); }
    auto data() const -> const std::vector<std::byte>& { return out_; }
    auto take_data() -> std::vector<std::byte> { return std::move(out_); }
    auto size() const -> std::size_t { return out_.size(); }
    auto bit_count() const -> std::size_t {
        return out_.size() * 8 + static_cast<std::size_t>(acc_bits_);
    }
    void clear() {
        out_.clear();
        acc_ = 0;
        acc_bits_ = 0;
    }

  private:
    std::vector<std::byte> out_;
    std::uint64_t acc_ = 0;
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
    std::uint64_t present_mask = 0;              // bit s set iff sym s (0..63) is present
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
inline auto fse_init_cstate2(const FseCTable& ct, int symbol) -> std::uint32_t {
    const std::uint32_t delta_nb_bits = ct.delta_nb_bits[static_cast<std::size_t>(symbol)];
    const std::uint32_t nb_bits_out = (delta_nb_bits + (1u << 15)) >> 16;
    std::uint32_t value = (nb_bits_out << 16) - delta_nb_bits;
    const int idx = static_cast<int>(value >> nb_bits_out) +
                    ct.delta_find_state[static_cast<std::size_t>(symbol)];
    return ct.state_table[static_cast<std::size_t>(idx)];
}

// FSE_encodeSymbol: emit `symbol` from `value` and advance the state.
inline void fse_encode_symbol(FseBitWriter& w, const FseCTable& ct,
                              std::uint32_t& value, int symbol) {
    const std::uint32_t delta_nb_bits = ct.delta_nb_bits[static_cast<std::size_t>(symbol)];
    const std::uint32_t nb_bits_out = (value + delta_nb_bits) >> 16;
    w.put_bits(value, static_cast<int>(nb_bits_out));
    const int idx = static_cast<int>(value >> nb_bits_out) +
                    ct.delta_find_state[static_cast<std::size_t>(symbol)];
    value = ct.state_table[static_cast<std::size_t>(idx)];
}

// FSE_flushCState: emit the final state (the decoder's initial state).
inline void fse_flush_cstate(FseBitWriter& w, const FseCTable& ct,
                             std::uint32_t value) {
    w.put_bits(value, ct.table_log);
}

// --- Shared FSE table helpers (NCount + normalization + decode table) ---

// Normalize frequencies to a distribution summing to 1<<table_log. Symbols
// with zero frequency get zero; present symbols get at least 1. No -1
// entries (useLowProbCount = 0, as zstd uses for Huffman weights).
void fse_normalize(const unsigned* freqs, int total, int max_symbol,
                   int table_log, std::vector<int>& norm);

// Variant that assigns -1 ("less than 1" probability) to rare symbols.
void fse_normalize_lowprob(const unsigned* freqs, int total, int max_symbol,
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
