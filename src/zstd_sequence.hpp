// fzip — zstd sequence decoder (RFC 8878 §4.2.2).
// Handles decoding and executing sequences in compressed blocks.
#pragma once

#include <cstdint>
#include <vector>

#include "zstd_fse.hpp"
#include "zstd_predefined.hpp"

namespace fzip::zstd {

// One decoded sequence.
struct Sequence {
    int literals_length;  // number of literal bytes to copy
    int match_length;     // number of match bytes to copy
    int offset;           // match distance (1-based)
};

// Repeat offset state (3 recent offsets, initially {1, 4, 8}).
struct RepeatOffsets {
    int offsets[3] = {1, 4, 8};
};

// Decode all sequences in a compressed block. The sequences use 3 interleaved
// FSE streams (literals length, offset code, match length). Each stream's
// table and accuracy log are provided by the caller (predefined, RLE, or a
// per-block FSE table). Returns the list of sequences.
auto decode_sequences(const std::byte* data, std::size_t size,
                      int num_sequences,
                      const FseSeqSymbol* litlen_table, int ll_acc,
                      const FseSeqSymbol* offset_table, int of_acc,
                      const FseSeqSymbol* matchlen_table, int ml_acc)
    -> std::vector<Sequence>;

// Execute a sequence: copy `literals_length` literal bytes, then copy
// `match_length` bytes from `offset` bytes back in the output.
// Returns the produced bytes (literals + match copies).
auto execute_sequences(const std::vector<Sequence>& sequences,
                       const std::vector<std::byte>& literals,
                       RepeatOffsets& repeat) -> std::vector<std::byte>;

// Length code tables (RFC 8878 §4.2.2).
// Map a length code (0..35 for litlen, 0..52 for matchlen) to:
//   - base length
//   - number of extra bits
auto litlen_code_to_base(int code) -> int;
auto litlen_code_to_extra(int code) -> int;
auto matchlen_code_to_base(int code) -> int;
auto matchlen_code_to_extra(int code) -> int;

// Offset code (0..31) to base distance.
auto offset_code_to_base(int code) -> int;

}  // namespace fzip::zstd
