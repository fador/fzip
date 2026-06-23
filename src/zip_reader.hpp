// fzip — ZIP container reader (for round-trip verification).
// Reads a ZIP archive produced by fzip, parses the central directory, and
// extracts entries using the appropriate codec.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "codec.hpp"

namespace fzip {

struct ZipReadEntry {
    std::string name;
    std::uint64_t uncompressed_size = 0;
    std::uint64_t compressed_size = 0;
    std::uint32_t crc32 = 0;
    std::uint16_t method = 0;  // compression method ID
    std::uint64_t local_header_offset = 0;
    CodecId codec = CodecId::Store;
};

// Read the central directory from a ZIP file and return the list of entries.
auto read_zip_entries(const std::string& archive_path)
    -> std::vector<ZipReadEntry>;

// Extract one entry from a ZIP file (reads compressed data from local header,
// decompresses, verifies CRC-32). Returns the uncompressed bytes.
auto extract_entry(const std::string& archive_path, const ZipReadEntry& entry)
    -> std::vector<std::byte>;

// Extract all entries from a ZIP file. Returns a map of name -> uncompressed
// bytes. Throws on CRC mismatch.
auto extract_all(const std::string& archive_path)
    -> std::vector<std::pair<std::string, std::vector<std::byte>>>;

}  // namespace fzip
