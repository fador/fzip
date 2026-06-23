// fzip — ZIP container writer.
//
// Hand-rolled writer for the ZIP/ZIP64 file format (PKWARE APPNOTE 6.3.10).
// Emits: local file headers (PK\x03\x04), file data, central directory
// (PK\x01\x02), and end-of-central-directory record (PK\x05\x06). ZIP64
// extensions (extra field 0x0001, EOCD64, locator) are emitted when any
// entry exceeds 4 GiB, the archive exceeds 4 GiB, or there are more than
// 65535 entries.
//
// Stage 1: Store method (compression method 0) only.
// Stage 2: ZIP64 + extra fields + alignment.
// Stage 3+: pluggable codecs via CodecId.
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "codec.hpp"

namespace fzip {

// Metadata describing one entry to be written into the archive.
struct ZipEntry {
    std::string name;             // entry name (stored as UTF-8)
    std::vector<std::byte> data;  // uncompressed content
    std::uint16_t mod_time = 0;   // DOS time
    std::uint16_t mod_date = 0;   // DOS date
    std::uint64_t mtime_unix = 0; // seconds since 1970-01-01 (for UT extra)
};

// Write a ZIP archive containing `entries` to `archive_path`.
// `codec` selects the compression method; `level` is codec-specific.
// Returns true on success, false on I/O error.
auto write_zip(const std::string& archive_path,
               const std::vector<ZipEntry>& entries,
               CodecId codec = CodecId::Store,
               int level = 6) -> bool;

// Convenience: read each file from disk and store it (method 0).
// Filenames in the archive are the basenames of the input paths.
auto write_store_zip(const std::string& archive_path,
                     const std::vector<std::string>& file_paths) -> bool;

// Convert a Unix time (seconds since epoch) to DOS time/date fields.
auto dos_time_date(std::uint64_t unix_seconds)
    -> std::pair<std::uint16_t, std::uint16_t>;

}  // namespace fzip
