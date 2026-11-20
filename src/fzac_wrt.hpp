// fzip — Dynamic Universal Word & Structural Token Replacement Transform (WRT)
// for FZAC v3. 100% data-driven and bijective on arbitrary byte streams.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fzip::fzac_v3 {

struct WrtEncoded {
    std::vector<std::byte> dict_blob;  // Front-coded serialized dictionary (to be compressed)
    std::vector<std::uint8_t> stream;  // Transformed token stream
    bool beneficial = false;           // True if WRT achieved substantial shrinkage
};

// Analyze `raw[0..n-1]`, build a dynamic frequency-tiered vocabulary of words
// (with case folding) and structural/markup tokens, and transform `raw`.
// Sets `beneficial = false` if the input is not text-like or doesn't shrink enough.
auto wrt_forward(const std::uint8_t* raw, std::size_t n) -> WrtEncoded;

// Reconstruct exact original bytes `dst[0..raw_size-1]` from `dict_blob` and `stream`.
// Throws `std::runtime_error` on malformed input.
void wrt_inverse(const std::uint8_t* dict_blob, std::size_t dict_len,
                 const std::uint8_t* stream, std::size_t stream_len,
                 std::uint8_t* dst, std::size_t raw_size);

}  // namespace fzip::fzac_v3
