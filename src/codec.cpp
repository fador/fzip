// fzip — Codec implementation: dispatch to Store / Deflate / Zstd.
#include "codec.hpp"

#include <span>
#include <stdexcept>

#include "deflate.hpp"
#include "file_type.hpp"
#include "zstd_codec.hpp"

namespace fzip {

namespace {

auto compress_with(CodecId codec, std::span<const std::byte> data,
                   int level, [[maybe_unused]] std::string_view hint_path) -> CompressedEntry {
    if (data.empty()) {
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
                   std::string_view hint_path) -> CompressedEntry {
    auto ftype = detect_file_type(hint_path, data);

    switch (ftype) {
        case FileType::Incompressible:
            // Already compressed — Store.
            return compress_with(CodecId::Store, data, 0, hint_path);

        case FileType::Executable:
            // Executables: deflate-9 until the custom zstd compressor emits
            // compressed blocks (it currently only writes raw blocks and
            // falls back to Store, i.e. no compression at all).
            return compress_with(CodecId::Deflate, data, 9, hint_path);

        case FileType::Text:
            // Text/XML/JSON/source: deflate-9 until zstd compressed blocks are
            // available. Deflate is strictly better than Store here.
            return compress_with(CodecId::Deflate, data, 9, hint_path);

        case FileType::Binary:
        default:
            // General binary: deflate-6 (good universal default).
            return compress_with(CodecId::Deflate, data, 6, hint_path);
    }
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
