// fzip — Zstd codec implementation (Stage 4 stub).
#include "zstd_codec.hpp"

namespace fzip {

auto zstd_compress(std::span<const std::byte> /*data*/, int /*level*/,
                   int /*long_distance_log*/, int /*target_cblock_size*/)
    -> std::vector<std::byte> {
    return {};  // Implemented in Stage 4.
}

auto zstd_decompress(std::span<const std::byte> /*data*/,
                     std::size_t /*expected_size*/) -> std::vector<std::byte> {
    return {};  // Implemented in Stage 4.
}

}  // namespace fzip
