// fzip — ZIP container writer implementation.
#include "zip_writer.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <span>
#include <vector>

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
constexpr std::uint16_t kExtraZip64 = 0x0001u;  // ZIP64 extended information
constexpr std::uint16_t kExtraUT    = 0x5455u;  // "UT" Unix timestamp
constexpr std::uint16_t kExtraUnix  = 0x7875u;  // "ux" Unix UID/GID

// Data alignment target: entry data starts at a multiple of this many bytes.
// 4096 matches the page size and Android APK zipalign v2 requirements.
constexpr std::uint64_t kDataAlignment = 4096;

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
    // Unix extra metadata.
    std::uint64_t mtime_unix = 0;
    std::uint64_t atime_unix = 0;
    std::uint64_t ctime_unix = 0;
    std::uint16_t uid = 0;
    std::uint16_t gid = 0;
};

// A small helper to build up an extra-field block in memory before writing
// it into a header. Extra fields are a sequence of (HeaderID u16, Size u16,
// Data[Size]) records.
class ExtraFields {
  public:
    void add(std::uint16_t id, std::span<const std::byte> payload) {
        io::write_u16(buf_, id);
        io::write_u16(buf_, static_cast<std::uint16_t>(payload.size()));
        io::write_bytes(buf_, payload);
    }
    void add_u16(std::uint16_t id, std::uint16_t v) {
        std::byte b[2]{};
        b[0] = static_cast<std::byte>(v & 0xFFu);
        b[1] = static_cast<std::byte>((v >> 8) & 0xFFu);
        add(id, std::span<const std::byte>{b, 2});
    }
    void add_u32(std::uint16_t id, std::uint32_t v) {
        std::byte b[4]{};
        for (int i = 0; i < 4; ++i) {
            b[i] = static_cast<std::byte>((v >> (8 * i)) & 0xFFu);
        }
        add(id, std::span<const std::byte>{b, 4});
    }
    auto data() const -> std::span<const std::byte> {
        return std::span<const std::byte>{
            reinterpret_cast<const std::byte*>(buf_.str().data()),
            buf_.str().size()};
    }
    auto size() const -> std::uint16_t {
        return static_cast<std::uint16_t>(buf_.str().size());
    }

  private:
    // We serialize into a string-stream backed buffer.
    std::ostringstream buf_;
};

// Build the extra-fields block for a LOCAL file header.
// Local headers only carry the ZIP64 sizes (no offset needed there).
auto build_local_extra(const EntryRecord& e, bool include_ut_unix)
    -> std::vector<std::byte> {
    ExtraFields ef;
    if (e.zip64_entry) {
        std::byte payload[16]{};
        for (int i = 0; i < 8; ++i) {
            payload[i] = static_cast<std::byte>(
                (e.uncompressed_size >> (8 * i)) & 0xFFu);
            payload[8 + i] = static_cast<std::byte>(
                (e.compressed_size >> (8 * i)) & 0xFFu);
        }
        ef.add(kExtraZip64, std::span<const std::byte>{payload, 16});
    }
    if (include_ut_unix) {
        // UT extra: 1 byte flags + optional mtime/atime/ctime (u32 each).
        // flags bit 0 = mtime present, bit 1 = atime, bit 2 = ctime.
        std::byte payload[1 + 4 + 4 + 4]{};
        payload[0] = static_cast<std::byte>(0x07u);  // mtime+atime+ctime
        auto m = static_cast<std::uint32_t>(e.mtime_unix);
        auto a = static_cast<std::uint32_t>(e.atime_unix);
        auto c = static_cast<std::uint32_t>(e.ctime_unix);
        for (int i = 0; i < 4; ++i) {
            payload[1 + i]  = static_cast<std::byte>((m >> (8 * i)) & 0xFFu);
            payload[5 + i]  = static_cast<std::byte>((a >> (8 * i)) & 0xFFu);
            payload[9 + i]  = static_cast<std::byte>((c >> (8 * i)) & 0xFFu);
        }
        ef.add(kExtraUT, std::span<const std::byte>{payload, 13});
        // Unix "ux" extra: 1 byte version + 1 byte uid size + uid +
        // 1 byte gid size + gid. We use 2-byte uid/gid.
        std::byte ux[1 + 1 + 2 + 1 + 2]{};
        ux[0] = static_cast<std::byte>(1);  // version
        ux[1] = static_cast<std::byte>(2);  // uid size
        ux[2] = static_cast<std::byte>(e.uid & 0xFFu);
        ux[3] = static_cast<std::byte>((e.uid >> 8) & 0xFFu);
        ux[4] = static_cast<std::byte>(2);  // gid size
        ux[5] = static_cast<std::byte>(e.gid & 0xFFu);
        ux[6] = static_cast<std::byte>((e.gid >> 8) & 0xFFu);
        ef.add(kExtraUnix, std::span<const std::byte>{ux, 7});
    }
    auto d = ef.data();
    return std::vector<std::byte>{d.begin(), d.end()};
}

// Build the extra-fields block for a CENTRAL directory header.
// The central directory carries ZIP64 sizes+offset, and the UT/Unix fields
// (UT in central is the same content per APPNOTE; some writers omit atime/
// ctime in central — we keep them for simplicity).
auto build_central_extra(const EntryRecord& e, bool include_ut_unix)
    -> std::vector<std::byte> {
    ExtraFields ef;
    if (e.zip64_entry) {
        std::byte payload[24]{};
        for (int i = 0; i < 8; ++i) {
            payload[i] = static_cast<std::byte>(
                (e.uncompressed_size >> (8 * i)) & 0xFFu);
            payload[8 + i] = static_cast<std::byte>(
                (e.compressed_size >> (8 * i)) & 0xFFu);
            payload[16 + i] = static_cast<std::byte>(
                (e.local_header_offset >> (8 * i)) & 0xFFu);
        }
        ef.add(kExtraZip64, std::span<const std::byte>{payload, 24});
    }
    if (include_ut_unix) {
        std::byte payload[1 + 4 + 4 + 4]{};
        payload[0] = static_cast<std::byte>(0x07u);
        auto m = static_cast<std::uint32_t>(e.mtime_unix);
        auto a = static_cast<std::uint32_t>(e.atime_unix);
        auto c = static_cast<std::uint32_t>(e.ctime_unix);
        for (int i = 0; i < 4; ++i) {
            payload[1 + i]  = static_cast<std::byte>((m >> (8 * i)) & 0xFFu);
            payload[5 + i]  = static_cast<std::byte>((a >> (8 * i)) & 0xFFu);
            payload[9 + i]  = static_cast<std::byte>((c >> (8 * i)) & 0xFFu);
        }
        ef.add(kExtraUT, std::span<const std::byte>{payload, 13});
        std::byte ux[1 + 1 + 2 + 1 + 2]{};
        ux[0] = static_cast<std::byte>(1);
        ux[1] = static_cast<std::byte>(2);
        ux[2] = static_cast<std::byte>(e.uid & 0xFFu);
        ux[3] = static_cast<std::byte>((e.uid >> 8) & 0xFFu);
        ux[4] = static_cast<std::byte>(2);
        ux[5] = static_cast<std::byte>(e.gid & 0xFFu);
        ux[6] = static_cast<std::byte>((e.gid >> 8) & 0xFFu);
        ef.add(kExtraUnix, std::span<const std::byte>{ux, 7});
    }
    auto d = ef.data();
    return std::vector<std::byte>{d.begin(), d.end()};
}

// Write the local file header for an entry. `extra` is the pre-built extra
// field block. Returns the number of bytes written (30 + name + extra).
auto write_local_header(std::ostream& out, const EntryRecord& e,
                        std::span<const std::byte> extra) -> std::uint64_t {
    io::write_u32(out, kSigLocalHeader);
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
    io::write_u16(out, static_cast<std::uint16_t>(extra.size()));
    io::write_string(out, e.name);
    io::write_bytes(out, extra);
    return 30 + e.name.size() + extra.size();
}

// Write one central-directory header.
auto write_central_header(std::ostream& out, const EntryRecord& e,
                          std::span<const std::byte> extra) {
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
    io::write_u16(out, static_cast<std::uint16_t>(extra.size()));
    io::write_u16(out, 0);  // comment length
    io::write_u16(out, 0);  // disk number start
    io::write_u16(out, 0);  // internal attributes
    io::write_u32(out, 0);  // external attributes (0 = normal)
    io::write_u32(out, e.zip64_entry && e.local_header_offset >= k32Sentinel
                         ? k32Sentinel
                         : static_cast<std::uint32_t>(e.local_header_offset));
    io::write_string(out, e.name);
    io::write_bytes(out, extra);
}

// Write padding bytes to align the next write to a multiple of
// `kDataAlignment` from the start of the archive. We insert a dummy extra
// field (header 0x0000, zero-payload) sized to reach the target offset.
// Because the local header itself is variable-length, we compute the padding
// *after* writing the real extra fields, by extending the extra block.
auto write_alignment_padding(std::ostream& out, std::uint64_t current_offset)
    -> std::uint64_t {
    if (current_offset % kDataAlignment == 0) return 0;
    std::uint64_t pad = kDataAlignment - (current_offset % kDataAlignment);
    // We must write pad bytes of raw padding. We use a "data descriptor"-
    // free approach: insert raw zero bytes between entries. This is legal
    // because the central directory records the actual local header offset,
    // and readers scan via the central directory, not sequentially.
    std::string zeros(pad, '\0');
    out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    return pad;
}

// Write the end-of-central-directory record, emitting ZIP64 records when
// the archive exceeds classic limits.
void write_eocd(std::ostream& out, std::uint64_t cd_size,
                std::uint64_t cd_offset, std::uint64_t entry_count) {
    const bool zip64 = entry_count >= k16Sentinel ||
                       cd_size >= k32Sentinel ||
                       cd_offset >= k32Sentinel;

    if (zip64) {
        // EOCD64 record. The "size of remaining record" field counts the
        // bytes after the 12-byte (sig + size) header: vmade(2) + vneed(2)
        // + disk(4) + cd_disk(4) + n_disk(8) + n_total(8) + cd_size(8) +
        // cd_off(8) = 44.
        io::write_u32(out, 0x06064b50u);
        io::write_u64(out, 44);  // size of remaining EOCD64 (fixed part)
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
        r.uncompressed_size = entry.data.size();
        r.crc32 = crc32(std::span<const std::byte>{entry.data});
        r.mtime_unix = entry.mtime_unix;
        r.atime_unix = entry.mtime_unix;  // we only track mtime
        r.ctime_unix = entry.mtime_unix;
        r.uid = 0;
        r.gid = 0;

        // Compress (or store) the payload.
        CompressedEntry ce = compress(codec, std::span<const std::byte>{entry.data},
                                      level, entry.name);
        r.method = static_cast<std::uint16_t>(ce.codec);
        r.compressed_size = ce.data.size();
        r.zip64_entry = needs_zip64_size(r.uncompressed_size, r.compressed_size);

        // Build the local extra fields (ZIP64 + UT + Unix).
        auto local_extra = build_local_extra(r, /*include_ut_unix=*/true);

        // Align entry data to kDataAlignment for mmap-friendly extraction
        // (zipalign-compatible). Only align entries whose uncompressed size
        // is at least the alignment boundary; for tiny files the padding
        // overhead would dominate.
        const std::uint64_t header_size = 30 + r.name.size() + local_extra.size();
        const bool align = r.uncompressed_size >= kDataAlignment;
        std::uint64_t data_start = offset + header_size;
        if (align && data_start % kDataAlignment != 0) {
            std::uint64_t pad = kDataAlignment - (data_start % kDataAlignment);
            // Insert pad zero bytes before the local header.
            std::string zeros(pad, '\0');
            out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
            offset += pad;
        }

        r.local_header_offset = offset;
        write_local_header(out, r, local_extra);
        io::write_bytes(out, std::span<const std::byte>{ce.data});

        offset = r.local_header_offset + header_size + r.compressed_size;
        records.push_back(std::move(r));
    }

    // Central directory.
    std::uint64_t cd_offset = offset;
    for (const auto& r : records) {
        auto central_extra = build_central_extra(r, /*include_ut_unix=*/true);
        write_central_header(out, r, central_extra);
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

    for (const auto& path : file_paths) {
        fs::path p(path);
        ZipEntry e;
        e.name = p.filename().string();
        e.data = io::read_file(path);
        // Convert fs::last_write_time (file_clock) to Unix seconds.
        // file_clock since C++20 is the same as system_clock on Windows
        // MSVC, but to be portable we go through to_time_t.
        auto ftime = fs::last_write_time(p);
        auto sctime = std::chrono::time_point_cast<std::chrono::seconds>(
            ftime);
        auto since = sctime.time_since_epoch();
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(since);
        e.mtime_unix = static_cast<std::uint64_t>(secs.count());
        auto [dos_t, dos_d] = dos_time_date(e.mtime_unix);
        e.mod_time = dos_t;
        e.mod_date = dos_d;
        entries.push_back(std::move(e));
    }

    return write_zip(archive_path, entries, CodecId::Store, 0);
}

}  // namespace fzip
