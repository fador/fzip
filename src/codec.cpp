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
                   std::string_view hint_path, int level) -> CompressedEntry {
    auto ftype = detect_file_type(hint_path, data);

    // Already-compressed formats never benefit (and Store is a hard cost
    // floor for everything else). Skip the trial pass entirely.
    if (ftype == FileType::Incompressible) {
        return compress_with(CodecId::Store, data, 0, hint_path);
    }

    // Trial-compress the candidates and keep the smallest payload. Store is
    // the guaranteed bound, so the result is never worse than either codec.
    // Deflate at its top level handles small/structured data well; zstd wins
    // on larger or more redundant inputs.
    CompressedEntry best = compress_with(CodecId::Store, data, 0, hint_path);

    auto consider = [&](CompressedEntry&& cand) {
        if (cand.data.size() < best.data.size()) {
            best = std::move(cand);
        }
    };

    consider(compress_with(CodecId::Deflate, data, 9, hint_path));
    consider(compress_with(CodecId::Zstd, data, level, hint_path));

    return best;
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
