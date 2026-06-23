// fzip — Codec implementation (Stage 5 stub).
#include "codec.hpp"

namespace fzip {

auto compress(CodecId /*codec*/, std::span<const std::byte> data,
              int /*level*/, std::string_view /*hint_path*/)
    -> CompressedEntry {
    CompressedEntry e;
    e.codec = CodecId::Store;
    e.data.assign(data.begin(), data.end());
    return e;
}

auto compress_auto(std::span<const std::byte> data,
                   std::string_view /*hint_path*/) -> CompressedEntry {
    CompressedEntry e;
    e.codec = CodecId::Store;
    e.data.assign(data.begin(), data.end());
    return e;
}

auto decompress(const CompressedEntry& entry) -> std::vector<std::byte> {
    return entry.data;  // Only correct for Store until Stage 4 is implemented.
}

}  // namespace fzip
