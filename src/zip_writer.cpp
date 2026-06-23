// fzip — ZIP container writer implementation.
#include "zip_writer.hpp"

#include <algorithm>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <span>

#include "codec.hpp"
#include "crc32.hpp"
#include "io.hpp"

namespace fzip {

namespace {

// ZIP signatures.
constexpr std::uint32_t kSigLocalHeader   = 0x04034b50u;
constexpr std::uint32_t kSigCentralHeader = 0x02014b50u;
constexpr std::uint32_t kSigEocd          = 0x06054b50u;

// Compression method IDs (stored in headers).
constexpr std::uint16_t kMethodStore   = 0;
constexpr std::uint16_t kMethodDeflate = 8;

// General-purpose bit flags.
constexpr std::uint16_t kFlagUtf8 = 0x0800u;  // bit 11: filename is UTF-8

// Smallest version needed to extract for classic ZIP (Store/Deflate).
constexpr std::uint16_t kVersionClassic = 20;  // 2.0
// Version needed for ZIP64.
constexpr std::uint16_t kVersionZip64 = 45;    // 4.5

// Sentinel values that signal "real value is in the ZIP64 extra field".
constexpr std::uint32_t k32Sentinel = 0xFFFFFFFFu;
constexpr std::uint16_t k16Sentinel = 0xFFFFu;

// Extra-field header IDs.
constexpr std::uint16_t kExtraZip64 = 0x0001u;

// Decide whether an entry requires ZIP64 based on its (un)compressed size.
auto needs_zip64_size(std::uint64_t uncompressed,
                      std::uint64_t compressed) -> bool {
    return uncompressed >= k32Sentinel || compressed >= k32Sentinel;
}

// Record kept while streaming entries so we can write the central directory
// after all local headers + data are on disk.
struct EntryRecord {
    std::string name;
    std::uint16_t mod_time = 0;
    std::uint16_t mod_date = 0;
    std::uint16_t method = kMethodStore;
    std::uint16_t flags = kFlagUtf8;
    std::uint32_t crc32 = 0;
    std::uint64_t compressed_size = 0;
    std::uint64_t uncompressed_size = 0;
    std::uint64_t local_header_offset = 0;
    bool zip64_entry = false;  // this entry uses the ZIP64 extra field
};

// Write the local file header for an entry. Returns the number of bytes
// written (header + name + extra), so the caller knows where data starts.
auto write_local_header(std::ostream& out, const EntryRecord& e)
    -> std::uint64_t {
    // ZIP64 extra field payload: only the fields whose 32-bit counterpart
    // would overflow are included, in the order: uncompressed, compressed.
    std::uint16_t extra_size = 0;
    if (e.zip64_entry) {
        extra_size = 16;  // two 8-byte fields
    }

    io::write_u32(out, kSigLocalHeader);
    io::write_u16(out, e.zip64_entry ? kVersionZip64 : kVersionClassic);
    io::write_u16(out, e.flags);
    io::write_u16(out, e.method);
    io::write_u16(out, e.mod_time);
    io::write_u16(out, e.mod_date);
    io::write_u32(out, e.crc32);
    // 32-bit sizes: sentinel if ZIP64 extra holds them.
    io::write_u32(out, e.zip64_entry ? k32Sentinel
                                     : static_cast<std::uint32_t>(e.compressed_size));
    io::write_u32(out, e.zip64_entry ? k32Sentinel
                                     : static_cast<std::uint32_t>(e.uncompressed_size));
    io::write_u16(out, static_cast<std::uint16_t>(e.name.size()));
    io::write_u16(out, extra_size);
    io::write_string(out, e.name);

    if (e.zip64_entry) {
        // Header ID + size + two 8-byte values.
        io::write_u16(out, kExtraZip64);
        io::write_u16(out, extra_size - 4);
        io::write_u64(out, e.uncompressed_size);
        io::write_u64(out, e.compressed_size);
    }

    return 30 + e.name.size() + extra_size;
}

// Write one central-directory header.
auto write_central_header(std::ostream& out, const EntryRecord& e) {
    // The central directory always carries the ZIP64 extra for entries that
    // need it (and may carry offset/disk fields too).
    std::uint16_t extra_size = 0;
    if (e.zip64_entry) {
        // uncompressed + compressed + offset = 24 bytes payload.
        extra_size = 20;
    }

    io::write_u32(out, kSigCentralHeader);
    io::write_u16(out, 0x031Eu);  // version made by: UNIX (3) + 30 (3.0)
    io::write_u16(out, e.zip64_entry ? kVersionZip64 : kVersionClassic);
    io::write_u16(out, e.flags);
    io::write_u16(out, e.method);
    io::write_u16(out, e.mod_time);
    io::write_u16(out, e.mod_date);
    io::write_u32(out, e.crc32);
    io::write_u32(out, e.zip64_entry ? k32Sentinel
                                     : static_cast<std::uint32_t>(e.compressed_size));
    io::write_u32(out, e.zip64_entry ? k32Sentinel
                                     : static_cast<std::uint32_t>(e.uncompressed_size));
    io::write_u16(out, static_cast<std::uint16_t>(e.name.size()));
    io::write_u16(out, extra_size);
    io::write_u16(out, 0);  // comment length
    io::write_u16(out, 0);  // disk number start (0 unless ZIP64 overflows)
    io::write_u16(out, 0);  // internal attributes
    io::write_u32(out, 0);  // external attributes (0 = normal)
    io::write_u32(out, e.zip64_entry && e.local_header_offset >= k32Sentinel
                         ? k32Sentinel
                         : static_cast<std::uint32_t>(e.local_header_offset));
    io::write_string(out, e.name);

    if (e.zip64_entry) {
        io::write_u16(out, kExtraZip64);
        io::write_u16(out, extra_size - 4);  // 16
        io::write_u64(out, e.uncompressed_size);
        io::write_u64(out, e.compressed_size);
        io::write_u64(out, e.local_header_offset);
    }
}

// Write the end-of-central-directory record, emitting ZIP64 records when
// the archive exceeds classic limits.
auto write_eocd(std::ostream& out, std::uint64_t cd_size,
                std::uint64_t cd_offset, std::uint64_t entry_count)
    -> std::uint64_t {
    const bool zip64 = entry_count >= k16Sentinel ||
                       cd_size >= k32Sentinel ||
                       cd_offset >= k32Sentinel;

    if (zip64) {
        // EOCD64 record.
        io::write_u32(out, 0x06064b50u);
        io::write_u64(out, 56);  // size of remaining EOCD64 (fixed part)
        io::write_u16(out, 0x031Eu);  // version made by
        io::write_u16(out, kVersionZip64);
        io::write_u32(out, 0);  // this disk number
        io::write_u32(out, 0);  // disk with CD start
        io::write_u64(out, entry_count);
        io::write_u64(out, entry_count);
        io::write_u64(out, cd_size);
        io::write_u64(out, cd_offset);

        // EOCD64 locator.
        io::write_u32(out, 0x07064b50u);
        io::write_u32(out, 0);  // disk with EOCD64
        io::write_u64(out, cd_offset + cd_size);  // offset of EOCD64
        io::write_u32(out, 1);  // total disks
    }

    // Classic EOCD (always written; sentinels if ZIP64).
    io::write_u32(out, kSigEocd);
    io::write_u16(out, 0);  // this disk
    io::write_u16(out, 0);  // disk with CD
    io::write_u16(out, zip64 ? k16Sentinel
                             : static_cast<std::uint16_t>(entry_count));
    io::write_u16(out, zip64 ? k16Sentinel
                             : static_cast<std::uint16_t>(entry_count));
    io::write_u32(out, zip64 ? k32Sentinel
                             : static_cast<std::uint32_t>(cd_size));
    io::write_u32(out, zip64 ? k32Sentinel
                             : static_cast<std::uint32_t>(cd_offset));
    io::write_u16(out, 0);  // comment length
    return 0;
}

}  // namespace

auto dos_time_date(std::uint64_t unix_seconds)
    -> std::pair<std::uint16_t, std::uint16_t> {
    std::time_t t = static_cast<std::time_t>(unix_seconds);
    std::tm tm{};
    const std::pair<std::uint16_t, std::uint16_t> epoch{
        std::uint16_t{0}, std::uint16_t{0x21}};
#if defined(_WIN32)
    if (gmtime_s(&tm, &t) != 0) return epoch;  // 1980-01-01
#else
    if (gmtime_r(&t, &tm) == nullptr) return epoch;
#endif
    if (tm.tm_year < 80) return epoch;  // before 1980, clamp
    std::uint16_t dos_time = static_cast<std::uint16_t>(
        (tm.tm_hour << 11) | (tm.tm_min << 5) | (tm.tm_sec / 2));
    std::uint16_t dos_date = static_cast<std::uint16_t>(
        (((tm.tm_year + 1900) - 1980) << 9) | ((tm.tm_mon + 1) << 5) |
        tm.tm_mday);
    return std::pair<std::uint16_t, std::uint16_t>{dos_time, dos_date};
}

auto write_zip(const std::string& archive_path,
               const std::vector<ZipEntry>& entries,
               CodecId codec, int level) -> bool {
    std::ofstream out(archive_path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    std::vector<EntryRecord> records;
    records.reserve(entries.size());

    std::uint64_t offset = 0;

    for (const auto& entry : entries) {
        EntryRecord r;
        r.name = entry.name;
        r.mod_time = entry.mod_time;
        r.mod_date = entry.mod_date;
        r.flags = kFlagUtf8;
        r.local_header_offset = offset;
        r.uncompressed_size = entry.data.size();
        r.crc32 = crc32(std::span<const std::byte>{entry.data});

        // Compress (or store) the payload.
        CompressedEntry ce = compress(codec, std::span<const std::byte>{entry.data},
                                      level, entry.name);
        r.method = static_cast<std::uint16_t>(ce.codec);
        r.compressed_size = ce.data.size();
        r.zip64_entry = needs_zip64_size(r.uncompressed_size, r.compressed_size);

        write_local_header(out, r);
        io::write_bytes(out, std::span<const std::byte>{ce.data});

        offset = r.local_header_offset +
                 30 + r.name.size() +
                 (r.zip64_entry ? 20 : 0) +
                 r.compressed_size;
        records.push_back(std::move(r));
    }

    // Central directory.
    std::uint64_t cd_offset = offset;
    for (const auto& r : records) {
        write_central_header(out, r);
    }
    std::uint64_t cd_end = static_cast<std::uint64_t>(out.tellp());
    std::uint64_t cd_size = cd_end - cd_offset;

    write_eocd(out, cd_size, cd_offset, records.size());
    out.flush();
    return static_cast<bool>(out);
}

auto write_store_zip(const std::string& archive_path,
                     const std::vector<std::string>& file_paths) -> bool {
    namespace fs = std::filesystem;

    std::vector<ZipEntry> entries;
    entries.reserve(file_paths.size());

    std::time_t now = std::time(nullptr);
    auto [dos_t, dos_d] = dos_time_date(static_cast<std::uint64_t>(now));

    for (const auto& path : file_paths) {
        fs::path p(path);
        ZipEntry e;
        e.name = p.filename().string();
        e.data = io::read_file(path);
        e.mod_time = dos_t;
        e.mod_date = dos_d;
        e.mtime_unix = static_cast<std::uint64_t>(
            fs::last_write_time(p).time_since_epoch().count());
        // Use current time for DOS fields if file time is unusable; the
        // UT extra (Stage 2) will carry the real mtime.
        entries.push_back(std::move(e));
    }

    return write_zip(archive_path, entries, CodecId::Store, 0);
}

}  // namespace fzip
