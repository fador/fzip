// fzip — ZIP container writer implementation.
#include "zip_writer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <numeric>
#include <span>
#include <thread>
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
// Like zipalign, only *stored* entries are aligned: they are the only ones
// that can be mmapped and used in place. Padding compressed entries buys
// nothing and costs ~2 KB per entry on average.
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

// Byte-vector serialization helpers for ZIP headers.
template <std::size_t N>
inline void append_le(std::vector<std::byte>& out, std::uint64_t value) {
    static_assert(N >= 1 && N <= 8);
    const std::size_t old = out.size();
    out.resize(old + N);
    for (std::size_t i = 0; i < N; ++i) {
        out[old + i] = static_cast<std::byte>(value & 0xFFu);
        value >>= 8;
    }
}
inline void append_u16(std::vector<std::byte>& out, std::uint16_t v) { append_le<2>(out, v); }
inline void append_u32(std::vector<std::byte>& out, std::uint32_t v) { append_le<4>(out, v); }
inline void append_u64(std::vector<std::byte>& out, std::uint64_t v) { append_le<8>(out, v); }
inline void append_bytes(std::vector<std::byte>& out, std::span<const std::byte> data) {
    if (!data.empty()) {
        out.insert(out.end(), data.begin(), data.end());
    }
}
inline void append_string(std::vector<std::byte>& out, std::string_view s) {
    if (!s.empty()) {
        const auto* p = reinterpret_cast<const std::byte*>(s.data());
        out.insert(out.end(), p, p + s.size());
    }
}

// A small helper to build up an extra-field block in memory before writing
// it into a header. Extra fields are a sequence of (HeaderID u16, Size u16,
// Data[Size]) records.
class ExtraFields {
  public:
    void add(std::uint16_t id, std::span<const std::byte> payload) {
        append_u16(buf_, id);
        append_u16(buf_, static_cast<std::uint16_t>(payload.size()));
        append_bytes(buf_, payload);
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
        return std::span<const std::byte>{buf_.data(), buf_.size()};
    }
    auto size() const -> std::uint16_t {
        return static_cast<std::uint16_t>(buf_.size());
    }
    void clear() { buf_.clear(); }

  private:
    std::vector<std::byte> buf_;
};

// Append the UT (Unix timestamp) and, when meaningful, the "ux" (UID/GID)
// extra fields. Shared by local and central headers.
//
// UT: 1 byte flags + optional mtime/atime/ctime (u32 each); flags bit 0 =
// mtime, bit 1 = atime, bit 2 = ctime. atime/ctime are only written when
// they differ from mtime (they are usually copies), saving 8 bytes per
// header. The central-directory UT may only carry mtime per the Info-ZIP
// spec, while its flags mirror the local header.
void add_ut_unix(ExtraFields& ef, const EntryRecord& e, bool central) {
    const auto m = static_cast<std::uint32_t>(e.mtime_unix);
    const auto a = static_cast<std::uint32_t>(e.atime_unix);
    const auto c = static_cast<std::uint32_t>(e.ctime_unix);
    const bool has_a = (a != m);
    const bool has_c = (c != m);
    std::byte payload[1 + 4 + 4 + 4]{};
    payload[0] = static_cast<std::byte>(0x01u | (has_a ? 0x02u : 0u) |
                                        (has_c ? 0x04u : 0u));
    std::size_t n = 1;
    auto put32 = [&](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            payload[n++] = static_cast<std::byte>((v >> (8 * i)) & 0xFFu);
        }
    };
    put32(m);
    if (!central) {
        if (has_a) put32(a);
        if (has_c) put32(c);
    }
    ef.add(kExtraUT, std::span<const std::byte>{payload, n});

    // Unix "ux" extra: 1 byte version + 1 byte uid size + uid + 1 byte gid
    // size + gid (2-byte uid/gid). Omitted for uid = gid = 0: we don't
    // capture ownership, and recording root:root is both wasted bytes and
    // wrong when extracting as root.
    if (e.uid != 0 || e.gid != 0) {
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
}

// Build the extra-fields block for a LOCAL file header into `ef`.
// Local headers only carry the ZIP64 sizes (no offset needed there).
void build_local_extra(const EntryRecord& e, bool include_ut_unix,
                       ExtraFields& ef) {
    ef.clear();
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
    if (include_ut_unix) add_ut_unix(ef, e, /*central=*/false);
}

// Build the extra-fields block for a CENTRAL directory header into `ef`.
void build_central_extra(const EntryRecord& e, bool include_ut_unix,
                         ExtraFields& ef) {
    ef.clear();
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
    if (include_ut_unix) add_ut_unix(ef, e, /*central=*/true);
}

// Serialize and write the local file header for an entry using `buf`.
// Returns the number of bytes written (30 + name + extra).
auto write_local_header(std::ostream& out, const EntryRecord& e,
                        std::span<const std::byte> extra,
                        std::vector<std::byte>& buf) -> std::uint64_t {
    buf.clear();
    append_u32(buf, kSigLocalHeader);
    append_u16(buf, e.zip64_entry ? kVersionZip64 : kVersionClassic);
    append_u16(buf, e.flags);
    append_u16(buf, e.method);
    append_u16(buf, e.mod_time);
    append_u16(buf, e.mod_date);
    append_u32(buf, e.crc32);
    append_u32(buf, e.zip64_entry ? k32Sentinel
                                  : static_cast<std::uint32_t>(e.compressed_size));
    append_u32(buf, e.zip64_entry ? k32Sentinel
                                  : static_cast<std::uint32_t>(e.uncompressed_size));
    append_u16(buf, static_cast<std::uint16_t>(e.name.size()));
    append_u16(buf, static_cast<std::uint16_t>(extra.size()));
    append_string(buf, e.name);
    append_bytes(buf, extra);
    io::write_bytes(out, buf);
    return buf.size();
}

// Serialize and append one central-directory header into `buf`.
void append_central_header(std::vector<std::byte>& buf, const EntryRecord& e,
                           std::span<const std::byte> extra) {
    append_u32(buf, kSigCentralHeader);
    append_u16(buf, 0x031Eu);  // version made by: UNIX (3) + 30 (3.0)
    append_u16(buf, e.zip64_entry ? kVersionZip64 : kVersionClassic);
    append_u16(buf, e.flags);
    append_u16(buf, e.method);
    append_u16(buf, e.mod_time);
    append_u16(buf, e.mod_date);
    append_u32(buf, e.crc32);
    append_u32(buf, e.zip64_entry ? k32Sentinel
                                  : static_cast<std::uint32_t>(e.compressed_size));
    append_u32(buf, e.zip64_entry ? k32Sentinel
                                  : static_cast<std::uint32_t>(e.uncompressed_size));
    append_u16(buf, static_cast<std::uint16_t>(e.name.size()));
    append_u16(buf, static_cast<std::uint16_t>(extra.size()));
    append_u16(buf, 0);  // comment length
    append_u16(buf, 0);  // disk number start
    append_u16(buf, 0);  // internal attributes
    append_u32(buf, 0);  // external attributes (0 = normal)
    append_u32(buf, e.zip64_entry && e.local_header_offset >= k32Sentinel
                        ? k32Sentinel
                        : static_cast<std::uint32_t>(e.local_header_offset));
    append_string(buf, e.name);
    append_bytes(buf, extra);
}

// Write the end-of-central-directory record, emitting ZIP64 records when
// the archive exceeds classic limits.
void write_eocd(std::ostream& out, std::uint64_t cd_size,
                std::uint64_t cd_offset, std::uint64_t entry_count,
                std::vector<std::byte>& buf) {
    buf.clear();
    const bool zip64 = entry_count >= k16Sentinel ||
                       cd_size >= k32Sentinel ||
                       cd_offset >= k32Sentinel;

    if (zip64) {
        // EOCD64 record.
        append_u32(buf, 0x06064b50u);
        append_u64(buf, 44);  // size of remaining EOCD64 (fixed part)
        append_u16(buf, 0x031Eu);  // version made by
        append_u16(buf, kVersionZip64);
        append_u32(buf, 0);  // this disk number
        append_u32(buf, 0);  // disk with CD start
        append_u64(buf, entry_count);
        append_u64(buf, entry_count);
        append_u64(buf, cd_size);
        append_u64(buf, cd_offset);

        // EOCD64 locator.
        append_u32(buf, 0x07064b50u);
        append_u32(buf, 0);  // disk with EOCD64
        append_u64(buf, cd_offset + cd_size);  // offset of EOCD64
        append_u32(buf, 1);  // total disks
    }

    // Classic EOCD (always written; sentinels if ZIP64).
    append_u32(buf, kSigEocd);
    append_u16(buf, 0);  // this disk
    append_u16(buf, 0);  // disk with CD
    append_u16(buf, zip64 ? k16Sentinel
                          : static_cast<std::uint16_t>(entry_count));
    append_u16(buf, zip64 ? k16Sentinel
                          : static_cast<std::uint16_t>(entry_count));
    append_u32(buf, zip64 ? k32Sentinel
                          : static_cast<std::uint32_t>(cd_size));
    append_u32(buf, zip64 ? k32Sentinel
                          : static_cast<std::uint32_t>(cd_offset));
    append_u16(buf, 0);  // comment length
    io::write_bytes(out, buf);
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

namespace {

// Shared implementation of write_zip / write_zip_auto. Entries are
// independent, so they are compressed first (spread across the available
// cores) and then streamed to disk in order, keeping the archive layout
// deterministic.
template <typename CompressFn>
auto write_zip_impl(const std::string& archive_path,
                    const std::vector<ZipEntry>& entries,
                    CompressFn&& compress_one,
                    ProgressCallback on_progress) -> bool {
    const std::size_t n = entries.size();

    // --- Compress all entries (parallel where beneficial) ---
    std::vector<std::uint32_t> crcs(n, 0);
    std::vector<CompressedEntry> compressed(n);
    std::atomic<bool> cancelled{false};
    std::atomic<std::size_t> completed_count{0};
    std::mutex progress_mutex;

    auto do_one = [&](std::size_t i) {
        if (cancelled.load(std::memory_order_relaxed)) return;
        const auto& e = entries[i];
        crcs[i] = crc32(std::span<const std::byte>{e.data});
        compressed[i] = compress_one(std::span<const std::byte>{e.data},
                                     std::string_view{e.name});

        std::size_t done = completed_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (on_progress) {
            std::lock_guard<std::mutex> lock(progress_mutex);
            if (!cancelled.load(std::memory_order_relaxed)) {
                const std::uint64_t comp_sz =
                    (compressed[i].codec == CodecId::Store &&
                     compressed[i].data.empty())
                        ? e.data.size()
                        : compressed[i].data.size();
                EntryReport rep{
                    .index = i,
                    .name = e.name,
                    .uncompressed_size = e.data.size(),
                    .compressed_size = comp_sz,
                    .codec = compressed[i].codec
                };
                if (!on_progress(done, n, rep)) {
                    cancelled.store(true, std::memory_order_relaxed);
                }
            }
        }
    };

    // Process largest entries first (LPT) to avoid long-tail worker stragglers.
    std::vector<std::size_t> work_order(n);
    std::iota(work_order.begin(), work_order.end(), 0);
    if (n > 1) {
        std::sort(work_order.begin(), work_order.end(),
                  [&](std::size_t a, std::size_t b) {
                      return entries[a].data.size() > entries[b].data.size();
                  });
    }

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (n < 2 || hw <= 1) {
        detail::ThreadBudgetGuard budget(hw);
        for (std::size_t idx = 0; idx < n; ++idx) {
            do_one(work_order[idx]);
            if (cancelled.load(std::memory_order_relaxed)) return false;
        }
    } else {
        const unsigned nw =
            std::min<unsigned>(hw, static_cast<unsigned>(n));
        const unsigned per_worker_budget = std::max(1u, hw / nw);
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> workers;
        workers.reserve(nw);
        for (unsigned t = 0; t < nw; ++t) {
            workers.emplace_back([&]() {
                detail::ThreadBudgetGuard budget(per_worker_budget);
                for (;;) {
                    if (cancelled.load(std::memory_order_relaxed)) break;
                    std::size_t pos = next.fetch_add(1, std::memory_order_relaxed);
                    if (pos >= n) break;
                    do_one(work_order[pos]);
                }
            });
        }
        for (auto& w : workers) w.join();
    }

    if (cancelled.load(std::memory_order_relaxed)) return false;

    // --- Write local headers + data + central directory, in order ---
    std::vector<char> io_buf(1 << 20);
    std::ofstream out;
    out.rdbuf()->pubsetbuf(io_buf.data(), static_cast<std::streamsize>(io_buf.size()));
    out.open(archive_path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    std::vector<EntryRecord> records;
    records.reserve(n);
    std::uint64_t offset = 0;
    ExtraFields ef;
    std::vector<std::byte> hdr_buf;
    hdr_buf.reserve(512);
    static constexpr char kZeros[kDataAlignment]{};

    for (std::size_t i = 0; i < n; ++i) {
        const auto& entry = entries[i];
        const CompressedEntry& ce = compressed[i];
        const bool store_direct =
            (ce.codec == CodecId::Store && ce.data.empty() && !entry.data.empty());
        const std::span<const std::byte> payload =
            store_direct ? std::span<const std::byte>{entry.data}
                         : std::span<const std::byte>{ce.data};

        EntryRecord r;
        r.name = entry.name;
        r.mod_time = entry.mod_time;
        r.mod_date = entry.mod_date;
        r.flags = kFlagUtf8;
        r.uncompressed_size = entry.data.size();
        r.crc32 = crcs[i];
        r.mtime_unix = entry.mtime_unix;
        r.atime_unix = entry.mtime_unix;
        r.ctime_unix = entry.mtime_unix;
        r.uid = 0;
        r.gid = 0;

        r.method = static_cast<std::uint16_t>(ce.codec);
        r.compressed_size = payload.size();
        r.zip64_entry = needs_zip64_size(r.uncompressed_size, r.compressed_size);

        build_local_extra(r, /*include_ut_unix=*/false, ef);

        const std::uint64_t header_size = 30 + r.name.size() + ef.size();
        const bool align = r.method == kMethodStore &&
                           r.uncompressed_size >= kDataAlignment;
        std::uint64_t data_start = offset + header_size;
        if (align && data_start % kDataAlignment != 0) {
            std::uint64_t pad = kDataAlignment - (data_start % kDataAlignment);
            out.write(kZeros, static_cast<std::streamsize>(pad));
            offset += pad;
        }

        r.local_header_offset = offset;
        write_local_header(out, r, ef.data(), hdr_buf);
        if (!payload.empty()) {
            io::write_bytes(out, payload);
        }

        offset = r.local_header_offset + header_size + r.compressed_size;
        records.push_back(std::move(r));
    }

    std::uint64_t cd_offset = offset;
    std::vector<std::byte> cd_buf;
    cd_buf.reserve(std::min<std::size_t>(n * 96, 1u << 20));
    std::uint64_t cd_size = 0;
    for (const auto& r : records) {
        build_central_extra(r, /*include_ut_unix=*/false, ef);
        append_central_header(cd_buf, r, ef.data());
        if (cd_buf.size() >= (1u << 19)) {
            cd_size += cd_buf.size();
            io::write_bytes(out, cd_buf);
            cd_buf.clear();
        }
    }
    if (!cd_buf.empty()) {
        cd_size += cd_buf.size();
        io::write_bytes(out, cd_buf);
    }

    write_eocd(out, cd_size, cd_offset, records.size(), hdr_buf);
    out.flush();
    return static_cast<bool>(out);
}

}  // namespace

auto write_zip(const std::string& archive_path,
               const std::vector<ZipEntry>& entries,
               CodecId codec, int level,
               ProgressCallback on_progress) -> bool {
    return write_zip_impl(
        archive_path, entries,
        [codec, level](std::span<const std::byte> data,
                       std::string_view name) {
            return detail::compress_for_writer(codec, data, level, name);
        },
        std::move(on_progress));
}

auto collect_entries(const std::vector<std::string>& paths)
    -> std::vector<ZipEntry> {
    namespace fs = std::filesystem;
    std::vector<ZipEntry> entries;

    auto add_file = [&](const fs::path& file_path, const std::string& entry_name) {
        ZipEntry e;
        e.name = entry_name;
        // Normalize any Windows backslashes in entry name to forward slashes
        for (char& c : e.name) {
            if (c == '\\') c = '/';
        }
        e.data = io::read_file(file_path.string());
        auto ftime = fs::last_write_time(file_path);
        auto sctime = std::chrono::time_point_cast<std::chrono::seconds>(ftime);
        auto since = sctime.time_since_epoch();
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(since);
        e.mtime_unix = static_cast<std::uint64_t>(secs.count());
        auto [dos_t, dos_d] = dos_time_date(e.mtime_unix);
        e.mod_time = dos_t;
        e.mod_date = dos_d;
        entries.push_back(std::move(e));
    };

    for (const auto& item : paths) {
        fs::path p(item);
        std::error_code ec;
        if (!fs::exists(p, ec)) continue;

        if (fs::is_directory(p, ec)) {
            // For a directory, the root component is the directory name itself
            fs::path base_parent = p.parent_path();
            for (const auto& dir_entry : fs::recursive_directory_iterator(p, ec)) {
                if (dir_entry.is_regular_file(ec)) {
                    fs::path rel = base_parent.empty() ? dir_entry.path() : fs::relative(dir_entry.path(), base_parent, ec);
                    add_file(dir_entry.path(), rel.generic_string());
                }
            }
        } else if (fs::is_regular_file(p, ec)) {
            add_file(p, p.filename().generic_string());
        }
    }
    return entries;
}

auto write_store_zip(const std::string& archive_path,
                     const std::vector<std::string>& file_paths) -> bool {
    auto entries = collect_entries(file_paths);
    return write_zip(archive_path, entries, CodecId::Store, 0);
}

auto write_zip_auto(const std::string& archive_path,
                    const std::vector<ZipEntry>& entries, int level,
                    ProgressCallback on_progress) -> bool {
    return write_zip_impl(
        archive_path, entries,
        [level](std::span<const std::byte> data, std::string_view name) {
            return detail::compress_auto_for_writer(data, name, level);
        },
        std::move(on_progress));
}

}  // namespace fzip
