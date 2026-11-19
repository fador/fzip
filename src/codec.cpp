// fzip — Codec implementation: dispatch to Store / Deflate / Zstd.
#include "codec.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <thread>

#include "deflate.hpp"
#include "file_type.hpp"
#include "zstd_codec.hpp"

namespace fzip {

namespace {

auto make_store_entry(std::span<const std::byte> data, bool omit_store_data)
    -> CompressedEntry {
    CompressedEntry e;
    e.codec = CodecId::Store;
    if (!omit_store_data && !data.empty()) {
        e.data.assign(data.begin(), data.end());
    }
    return e;
}

auto compress_with(CodecId codec, std::span<const std::byte> data,
                   int level, [[maybe_unused]] std::string_view hint_path,
                   bool omit_store_data) -> CompressedEntry {
    if (data.empty()) {
        return CompressedEntry{CodecId::Store, {}};
    }
    switch (codec) {
        case CodecId::Store:
            return make_store_entry(data, omit_store_data);
        case CodecId::Deflate: {
            auto out = deflate_compress(data, level);
            if (out.empty()) return make_store_entry(data, omit_store_data);
            return CompressedEntry{CodecId::Deflate, std::move(out)};
        }
        case CodecId::Zstd: {
            auto out = zstd_compress(data, level);
            if (out.empty()) return make_store_entry(data, omit_store_data);
            return CompressedEntry{CodecId::Zstd, std::move(out)};
        }
    }
    throw std::runtime_error("unknown codec");
}

auto compress_auto_impl(std::span<const std::byte> data,
                        std::string_view hint_path, int level,
                        bool omit_store_data) -> CompressedEntry {
    if (data.empty()) {
        return CompressedEntry{CodecId::Store, {}};
    }
    auto ftype = detect_file_type(hint_path, data);

    // Already-compressed formats never benefit (and Store is a hard cost
    // floor for everything else). Skip the trial pass entirely.
    if (ftype == FileType::Incompressible) {
        return make_store_entry(data, omit_store_data);
    }

    // Deflate-12 runs a deeper optimal parse; its cost is negligible on
    // small/medium entries, where deflate usually beats zstd anyway, so it
    // replaces level 9 there. Large entries keep level 9 to bound time.
    constexpr std::size_t kDeflate12Limit = 1u << 20;  // 1 MiB
    const int deflate_level = (data.size() <= kDeflate12Limit) ? 12 : 9;
    const int zstd_level = (level >= 19 && data.size() <= kDeflate12Limit) ? 22 : level;

    std::vector<std::byte> def_out;
    std::vector<std::byte> zst_out;

    const unsigned avail_threads = detail::effective_threads();
    if (avail_threads >= 2 && data.size() >= 65536) {
        const unsigned def_threads = std::max(1u, avail_threads / 2);
        const unsigned zst_threads = std::max(1u, avail_threads - def_threads);
        std::thread zst_thread([&]() {
            detail::ThreadBudgetGuard guard(zst_threads);
            zst_out = zstd_compress(data, zstd_level);
        });
        {
            detail::ThreadBudgetGuard guard(def_threads);
            def_out = deflate_compress(data, deflate_level);
        }
        zst_thread.join();
    } else {
        def_out = deflate_compress(data, deflate_level);
        zst_out = zstd_compress(data, zstd_level);
    }

    const std::size_t raw_size = data.size();
    const std::size_t def_size = def_out.empty() ? raw_size : def_out.size();
    const std::size_t zst_size = zst_out.empty() ? raw_size : zst_out.size();

    if (def_size < raw_size && def_size <= zst_size) {
        return CompressedEntry{CodecId::Deflate, std::move(def_out)};
    }
    if (zst_size < raw_size) {
        return CompressedEntry{CodecId::Zstd, std::move(zst_out)};
    }
    return make_store_entry(data, omit_store_data);
}

}  // namespace

namespace detail {

auto compress_for_writer(CodecId codec, std::span<const std::byte> data,
                         int level, std::string_view hint_path) -> CompressedEntry {
    return compress_with(codec, data, level, hint_path, /*omit_store_data=*/true);
}

auto compress_auto_for_writer(std::span<const std::byte> data,
                              std::string_view hint_path, int level)
    -> CompressedEntry {
    return compress_auto_impl(data, hint_path, level, /*omit_store_data=*/true);
}

auto decompress_span(CodecId codec, std::span<const std::byte> data,
                     std::size_t expected_size) -> std::vector<std::byte> {
    switch (codec) {
        case CodecId::Store:
            return std::vector<std::byte>(data.begin(), data.end());
        case CodecId::Deflate:
            return deflate_decompress(data, expected_size);
        case CodecId::Zstd:
            return zstd_decompress(data, expected_size);
    }
    throw std::runtime_error("unknown codec");
}

}  // namespace detail

auto compress(CodecId codec, std::span<const std::byte> data, int level,
              std::string_view hint_path) -> CompressedEntry {
    return compress_with(codec, data, level, hint_path, /*omit_store_data=*/false);
}

auto compress_auto(std::span<const std::byte> data,
                   std::string_view hint_path, int level) -> CompressedEntry {
    return compress_auto_impl(data, hint_path, level, /*omit_store_data=*/false);
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
