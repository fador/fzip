// fzip — File-type detection by magic bytes and extension. Stage 5.
#pragma once

#include <span>
#include <string_view>

namespace fzip {

enum class FileType {
    Text,            // plain text, source, XML, JSON, CSV, ...
    Executable,      // PE (Windows .exe/.dll), ELF, Mach-O
    Incompressible,  // already compressed: zip, gz, xz, zst, jpg, png, mp4, ...
    Binary,          // other binary (default for unrecognized)
};

// Classify `data` (first 512 bytes are enough) using magic bytes and the
// optional `path` extension as a hint.
auto detect_file_type(std::string_view path,
                      std::span<const std::byte> data) -> FileType;

}  // namespace fzip
