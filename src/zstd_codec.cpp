// fzip — Zstd codec (method 93) wrapper around custom zstd implementation.
#include "zstd_codec.hpp"

#include <stdexcept>
#include <vector>

#include "zstd.hpp"

namespace fzip {

auto zstd_compress(std::span<const std::byte> data, int level,
                   int long_distance_log, int /*target_cblock_size*/)
    -> std::vector<std::byte> {
    return zstd::compress(data, level, long_distance_log);
}

auto zstd_decompress(std::span<const std::byte> data,
                     std::size_t expected_size) -> std::vector<std::byte> {
    return zstd::decompress(data, expected_size);
}

}  // namespace fzip
