// fzip — zstd decompressor implementation (RFC 8878).
// Stage 0: stub that throws "not implemented". Will be filled in Stages 1-5.
#include "zstd.hpp"
#include "zstd_internal.hpp"

namespace fzip::zstd {

auto decompress(std::span<const std::byte> /*data*/,
                std::size_t /*expected_size*/) -> std::vector<std::byte> {
    throw ZstdError("zstd decompress: not implemented (Stage 0 stub)");
}

}  // namespace fzip::zstd
