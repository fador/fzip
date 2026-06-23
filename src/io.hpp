// fzip — small cross-platform binary I/O helpers for the ZIP writer.
// All multi-byte ZIP header fields are little-endian.
#pragma once

#include <cstdint>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fzip::io {

// Write a little-endian unsigned integer of N bytes to `out`.
template <std::size_t N>
void write_le(std::ostream& out, std::uint64_t value) {
    static_assert(N >= 1 && N <= 8);
    for (std::size_t i = 0; i < N; ++i) {
        auto byte = static_cast<char>(value & 0xFFu);
        out.put(byte);
        value >>= 8;
    }
}

inline void write_u16(std::ostream& out, std::uint16_t v) { write_le<2>(out, v); }
inline void write_u32(std::ostream& out, std::uint32_t v) { write_le<4>(out, v); }
inline void write_u64(std::ostream& out, std::uint64_t v) { write_le<8>(out, v); }

inline void write_bytes(std::ostream& out, std::span<const std::byte> data) {
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
}

inline void write_string(std::ostream& out, std::string_view s) {
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// Read an entire file into a byte vector. Throws on error.
inline auto read_file(const std::string& path) -> std::vector<std::byte> {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open file: " + path);
    }
    f.seekg(0, std::ios::end);
    auto end = f.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot size file: " + path);
    }
    f.seekg(0, std::ios::beg);
    std::vector<std::byte> buf(static_cast<std::size_t>(end));
    if (!buf.empty()) {
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(buf.size()));
        if (!f) {
            throw std::runtime_error("short read: " + path);
        }
    }
    return buf;
}

}  // namespace fzip::io
