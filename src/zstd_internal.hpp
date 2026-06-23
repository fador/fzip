// fzip — zstd internal types, bit I/O, and error codes.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace fzip::zstd {

// zstd magic number (little-endian).
constexpr std::uint32_t kMagic = 0xFD2FB528u;
// Skippable frame magic base (0x184D2A50 + N where N = 0..15).
constexpr std::uint32_t kSkippableMagicBase = 0x184D2A50u;

// Block types.
enum class BlockType : std::uint8_t {
    Raw = 0,
    RLE = 1,
    Compressed = 2,
    Reserved = 3,
};

// Error thrown on format violations.
struct ZstdError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// --- Bit reader (LSB-first, forward direction) ---
// zstd's main bitstream reads forward. FSE uses a reverse bitstream
// which is handled separately in zstd_fse.hpp.
class BitReader {
  public:
    explicit BitReader(const std::byte* data, std::size_t size)
        : data_(data), size_(size) {}

    auto read_bits(int n) -> std::uint64_t {
        std::uint64_t v = 0;
        for (int i = 0; i < n; ++i) {
            if (pos_ >= size_ * 8) throw ZstdError("bitreader: EOF");
            std::uint64_t bit = (static_cast<std::uint8_t>(data_[pos_ / 8])
                                 >> (pos_ % 8)) & 1u;
            v |= bit << i;
            ++pos_;
        }
        return v;
    }

    auto read_u8() -> std::uint8_t {
        return static_cast<std::uint8_t>(read_bits(8));
    }

    auto read_u32() -> std::uint32_t {
        return static_cast<std::uint32_t>(read_bits(32));
    }

    void skip_to_byte() {
        if (pos_ % 8 != 0) pos_ += 8 - (pos_ % 8);
    }

    auto bit_pos() const -> std::size_t { return pos_; }
    auto byte_pos() const -> std::size_t { return pos_ / 8; }
    auto remaining_bits() const -> std::size_t { return size_ * 8 - pos_; }

    void seek_bits(std::size_t bits) { pos_ += bits; }

  private:
    const std::byte* data_;
    std::size_t size_;
    std::size_t pos_ = 0;
};

}  // namespace fzip::zstd
