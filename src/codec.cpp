// fzip — Codec implementation: dispatch to Store / Deflate / Zstd.
#include "codec.hpp"

#include <span>
#include <stdexcept>

#include "deflate.hpp"
#include "zstd_codec.hpp"

namespace fzip {

namespace {

// Compress with a specific codec, falling back to Store if compression
// doesn't shrink the input or if the codec returns empty.
auto compress_with(CodecId codec, std::span<const std::byte> data,
                   int level, [[maybe_unused]] std::string_view hint_path) -> CompressedEntry {
    if (data.empty()) {
        // Empty input: Store is fine (zero-length entry).
        CompressedEntry e;
        e.codec = CodecId::Store;
        return e;
    }
    switch (codec) {
        case CodecId::Store: {
            CompressedEntry e;
            e.codec = CodecId::Store;
            e.data.assign(data.begin(), data.end());
            return e;
        }
        case CodecId::Deflate: {
            auto out = deflate_compress(data, level);
            if (out.empty()) {
                // Compression didn't help; fall back to Store.
                CompressedEntry e;
                e.codec = CodecId::Store;
                e.data.assign(data.begin(), data.end());
                return e;
            }
            CompressedEntry e;
            e.codec = CodecId::Deflate;
            e.data = std::move(out);
            return e;
        }
        case CodecId::Zstd: {
            auto out = zstd_compress(data, level);
            if (out.empty()) {
                CompressedEntry e;
                e.codec = CodecId::Store;
                e.data.assign(data.begin(), data.end());
                return e;
            }
            CompressedEntry e;
            e.codec = CodecId::Zstd;
            e.data = std::move(out);
            return e;
        }
    }
    throw std::runtime_error("unknown codec");
}

}  // namespace

auto compress(CodecId codec, std::span<const std::byte> data, int level,
              std::string_view hint_path) -> CompressedEntry {
    return compress_with(codec, data, level, hint_path);
}

auto compress_auto(std::span<const std::byte> data,
                   std::string_view /*hint_path*/) -> CompressedEntry {
    // Stage 5 will add type detection. For now, default to Deflate.
    return compress_with(CodecId::Deflate, data, 6, "");
}

auto decompress(const CompressedEntry& entry) -> std::vector<std::byte> {
    switch (entry.codec) {
        case CodecId::Store:
            return entry.data;
        case CodecId::Deflate:
            return deflate_decompress(entry.data, 0);
        case CodecId::Zstd:
            return zstd_decompress(entry.data, 0);
    }
    throw std::runtime_error("unknown codec");
}

}  // namespace fzip
