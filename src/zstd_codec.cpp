// fzip — Zstd codec (method 93) wrapper around libzstd 1.5.7.
#include "zstd_codec.hpp"

#include <stdexcept>
#include <vector>

#include <zstd.h>

namespace fzip {

auto zstd_compress(std::span<const std::byte> data, int level,
                   int long_distance_log, int target_cblock_size)
    -> std::vector<std::byte> {
    if (data.empty()) return {};

    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) throw std::runtime_error("ZSTD_createCCtx failed");

    // Set compression level (1-22; 22 = ultra).
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);

    // Long-distance matching: window log up to 27 (128 MiB).
    if (long_distance_log > 0) {
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_enableLongDistanceMatching, 1);
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_windowLog, long_distance_log);
    }

    // Target compressed block size for streaming decode friendliness.
    if (target_cblock_size > 0) {
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_targetCBlockSize,
                               target_cblock_size);
    }

    // Use 1 worker thread (we compress per-file, so parallelism is at the
    // archive level, not within a single entry).
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, 0);

    // Reserve output buffer: worst case is slightly larger than input for
    // incompressible data; ZSTD_compressBound gives the exact bound.
    std::size_t bound = ZSTD_compressBound(data.size());
    std::vector<std::byte> out(bound);

    std::size_t compressed = ZSTD_compress2(
        cctx,
        out.data(), out.size(),
        data.data(), data.size());

    ZSTD_freeCCtx(cctx);

    if (ZSTD_isError(compressed)) {
        // Compression error — signal Store fallback.
        return {};
    }

    out.resize(compressed);

    // If compression didn't shrink the input, signal Store.
    if (out.size() >= data.size()) return {};

    return out;
}

auto zstd_decompress(std::span<const std::byte> data,
                     std::size_t expected_size) -> std::vector<std::byte> {
    if (data.empty()) return {};

    // If expected_size is unknown (0), probe with ZSTD_getFrameContentSize.
    std::size_t dst_size = expected_size;
    if (dst_size == 0) {
        auto content = ZSTD_getFrameContentSize(data.data(), data.size());
        if (content == ZSTD_CONTENTSIZE_ERROR || content == ZSTD_CONTENTSIZE_UNKNOWN) {
            // Fallback: try a generous initial size and grow.
            dst_size = data.size() * 4;
        } else {
            dst_size = static_cast<std::size_t>(content);
        }
    }

    std::vector<std::byte> out(dst_size);
    std::size_t decompressed = ZSTD_decompress(
        out.data(), out.size(),
        data.data(), data.size());

    if (ZSTD_isError(decompressed)) {
        throw std::runtime_error(
            std::string("ZSTD_decompress failed: ") +
            ZSTD_getErrorName(decompressed));
    }

    out.resize(decompressed);
    return out;
}

}  // namespace fzip
