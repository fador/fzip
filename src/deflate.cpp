// fzip — DEFLATE encoder implementation (Stage 3 stub).
#include "deflate.hpp"

namespace fzip {

auto deflate_compress(std::span<const std::byte> /*data*/, int /*level*/)
    -> std::vector<std::byte> {
    return {};  // Implemented in Stage 3.
}

auto deflate_decompress(std::span<const std::byte> /*data*/,
                        std::size_t /*expected*/) -> std::vector<std::byte> {
    return {};  // Implemented in Stage 3 (test-only).
}

}  // namespace fzip
