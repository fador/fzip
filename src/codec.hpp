// fzip — Codec interface and per-file selector. Stage 5.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fzip {

enum class CodecId : std::uint16_t {
    Store = 0,
    Deflate = 8,
    Zstd = 93,
};

// Result of compressing one entry.
struct CompressedEntry {
    CodecId codec{CodecId::Store};
    std::vector<std::byte> data;
};

// Compress `data` using the selected `codec` and `level` (codec-specific).
// Returns an entry whose `codec` may differ from the request if Store is
// a better choice (e.g. compression failed to shrink the input).
auto compress(CodecId codec, std::span<const std::byte> data, int level,
              std::string_view hint_path) -> CompressedEntry;

// Auto-select a codec for `data` based on type detection and size.
auto compress_auto(std::span<const std::byte> data,
                   std::string_view hint_path) -> CompressedEntry;

// Decompress an entry back to its original bytes.
auto decompress(const CompressedEntry& entry) -> std::vector<std::byte>;

}  // namespace fzip
