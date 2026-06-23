// fzip — ZIP container reader implementation.
#include "zip_reader.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "codec.hpp"
#include "crc32.hpp"
#include "io.hpp"

namespace fzip {

namespace {

auto read_u16(const std::byte* p) -> std::uint16_t {
    return static_cast<std::uint16_t>(p[0]) |
           (static_cast<std::uint16_t>(p[1]) << 8);
}
auto read_u32(const std::byte* p) -> std::uint32_t {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}
auto read_u64(const std::byte* p) -> std::uint64_t {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (static_cast<std::uint64_t>(p[i]) << (8 * i));
    return v;
}

// Search backwards from `end` for the EOCD signature (PK\x05\x06).
auto find_eocd(const std::byte* data, std::size_t size) -> const std::byte* {
    // EOCD is at least 22 bytes; search from the end.
    if (size < 22) return nullptr;
    for (std::size_t i = size - 22; i > 0; --i) {
        if (read_u32(data + i) == 0x06054b50u) {
            return data + i;
        }
    }
    // Check offset 0.
    if (read_u32(data) == 0x06054b50u) return data;
    return nullptr;
}

// Parse one central-directory entry at `p`. Returns the number of bytes
// consumed (46 + name_len + extra_len + comment_len), or 0 on error.
auto parse_cd_entry(const std::byte* p, const std::byte* end,
                    ZipReadEntry& out) -> std::size_t {
    if (p + 46 > end) return 0;
    if (read_u32(p) != 0x02014b50u) return 0;  // not a CD header

    // Version made by (2), version needed (2), flags (2), method (2),
    // time (2), date (2), crc (4), comp_size (4), uncomp_size (4),
    // name_len (2), extra_len (2), comment_len (2), disk_start (2),
    // internal_attr (2), external_attr (4), local_header_offset (4).
    out.method = read_u16(p + 10);
    out.crc32 = read_u32(p + 16);
    out.compressed_size = read_u32(p + 20);
    out.uncompressed_size = read_u32(p + 24);
    auto name_len = read_u16(p + 28);
    auto extra_len = read_u16(p + 30);
    auto comment_len = read_u16(p + 32);
    out.local_header_offset = read_u32(p + 42);

    if (p + 46 + name_len > end) return 0;
    out.name = std::string(reinterpret_cast<const char*>(p + 46), name_len);

    // Check for ZIP64 extra field (0x0001) to get 64-bit sizes/offset.
    const std::byte* extra = p + 46 + name_len;
    const std::byte* extra_end = extra + extra_len;
    if (extra_end > end) extra_end = end;
    while (extra + 4 <= extra_end) {
        auto eid = read_u16(extra);
        auto esz = read_u16(extra + 2);
        if (extra + 4 + esz > extra_end) break;
        if (eid == 0x0001u) {
            const std::byte* ed = extra + 4;
            int off = 0;
            // ZIP64 extra: uncompressed (8), compressed (8), offset (8),
            // disk (4) — but only the fields whose 32-bit counterpart
            // was the sentinel 0xFFFFFFFF are present.
            if (out.uncompressed_size == 0xFFFFFFFFu && off + 8 <= esz) {
                out.uncompressed_size = read_u64(ed + off); off += 8;
            }
            if (out.compressed_size == 0xFFFFFFFFu && off + 8 <= esz) {
                out.compressed_size = read_u64(ed + off); off += 8;
            }
            if (out.local_header_offset == 0xFFFFFFFFu && off + 8 <= esz) {
                out.local_header_offset = read_u64(ed + off); off += 8;
            }
        }
        extra += 4 + esz;
    }

    // Map method ID to CodecId.
    switch (out.method) {
        case 0: out.codec = CodecId::Store; break;
        case 8: out.codec = CodecId::Deflate; break;
        case 93: out.codec = CodecId::Zstd; break;
        default: out.codec = CodecId::Store; break;  // unknown, try store
    }

    return 46 + name_len + extra_len + comment_len;
}

}  // namespace

auto read_zip_entries(const std::string& archive_path)
    -> std::vector<ZipReadEntry> {
    auto data = io::read_file(archive_path);
    const std::byte* base = data.data();
    const std::byte* end = base + data.size();

    const std::byte* eocd = find_eocd(base, data.size());
    if (!eocd) throw std::runtime_error("no EOCD found in " + archive_path);

    // Parse EOCD: get CD offset and size.
    auto cd_size_32 = read_u32(eocd + 12);
    auto cd_offset_32 = read_u32(eocd + 16);
    auto total_entries = read_u16(eocd + 10);

    std::uint64_t cd_offset = cd_offset_32;
    std::uint64_t cd_size = cd_size_32;
    std::uint64_t entry_count = total_entries;

    // If sentinels, look for ZIP64 EOCD64 (PK\x06\x06) which precedes EOCD.
    if (cd_offset_32 == 0xFFFFFFFFu || cd_size_32 == 0xFFFFFFFFu ||
        total_entries == 0xFFFFu) {
        // EOCD64 is at eocd64_off from the locator (PK\x06\x07) which is
        // immediately before the EOCD.
        if (eocd - 20 >= base && read_u32(eocd - 20) == 0x07064b50u) {
            auto eocd64_off = read_u64(eocd - 20 + 8);
            const std::byte* e64 = base + eocd64_off;
            if (e64 + 56 <= end && read_u32(e64) == 0x06064b50u) {
                entry_count = read_u64(e64 + 32);
                cd_size = read_u64(e64 + 40);
                cd_offset = read_u64(e64 + 48);
            }
        }
    }

    if (cd_offset + cd_size > data.size()) {
        throw std::runtime_error("CD offset out of bounds");
    }

    std::vector<ZipReadEntry> entries;
    const std::byte* p = base + cd_offset;
    const std::byte* cd_end = p + cd_size;
    for (std::uint64_t i = 0; i < entry_count && p < cd_end; ++i) {
        ZipReadEntry e;
        auto consumed = parse_cd_entry(p, cd_end, e);
        if (consumed == 0) break;
        entries.push_back(std::move(e));
        p += consumed;
    }
    return entries;
}

auto extract_entry(const std::string& archive_path, const ZipReadEntry& entry)
    -> std::vector<std::byte> {
    auto data = io::read_file(archive_path);
    const std::byte* base = data.data();

    // Read the local header at entry.local_header_offset to get the actual
    // compressed data offset (header may have different extra fields than CD).
    auto off = entry.local_header_offset;
    if (off + 30 > data.size()) {
        throw std::runtime_error("local header offset out of bounds");
    }
    const std::byte* lh = base + off;
    if (read_u32(lh) != 0x04034b50u) {
        throw std::runtime_error("bad local header signature");
    }
    auto name_len = read_u16(lh + 26);
    auto extra_len = read_u16(lh + 28);
    auto data_offset = off + 30 + name_len + extra_len;
    if (data_offset + entry.compressed_size > data.size()) {
        throw std::runtime_error("compressed data out of bounds");
    }

    std::span<const std::byte> comp_data{base + data_offset,
                                          entry.compressed_size};

    // Decompress.
    CompressedEntry ce;
    ce.codec = entry.codec;
    ce.data.assign(comp_data.begin(), comp_data.end());
    auto result = decompress(ce);

    // Verify CRC.
    auto actual_crc = crc32(std::span<const std::byte>{result.data(), result.size()});
    if (actual_crc != entry.crc32) {
        throw std::runtime_error("CRC mismatch: " + entry.name +
                                 " expected=" + std::to_string(entry.crc32) +
                                 " actual=" + std::to_string(actual_crc));
    }
    return result;
}

auto extract_all(const std::string& archive_path)
    -> std::vector<std::pair<std::string, std::vector<std::byte>>> {
    auto entries = read_zip_entries(archive_path);
    std::vector<std::pair<std::string, std::vector<std::byte>>> result;
    result.reserve(entries.size());
    for (const auto& e : entries) {
        auto data = extract_entry(archive_path, e);
        result.emplace_back(e.name, std::move(data));
    }
    return result;
}

}  // namespace fzip
