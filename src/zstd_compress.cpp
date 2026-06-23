// fzip — zstd compressor implementation (RFC 8878).
// Stage 0: stub that throws "not implemented". Will be filled in Stages 6-9.
#include "zstd.hpp"
#include "zstd_internal.hpp"

namespace fzip::zstd {

auto compress(std::span<const std::byte> /*data*/, int /*level*/,
              int /*long_distance_log*/) -> std::vector<std::byte> {
    throw ZstdError("zstd compress: not implemented (Stage 0 stub)");
}

}  // namespace fzip::zstd
