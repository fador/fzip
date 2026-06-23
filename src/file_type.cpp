// fzip — File-type detection by magic bytes and extension.
#include "file_type.hpp"

#include <algorithm>
#include <cstring>
#include <string>

namespace fzip {

namespace {

// Check if `data` starts with the given magic bytes.
auto starts_with(std::span<const std::byte> data,
                 const std::uint8_t* magic, std::size_t len) -> bool {
    if (data.size() < len) return false;
    return std::memcmp(data.data(), magic, len) == 0;
}

// Get the lowercase extension from a path.
auto extension_of(std::string_view path) -> std::string {
    auto pos = path.rfind('.');
    if (pos == std::string_view::npos) return {};
    std::string ext(path.substr(pos));
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

}  // namespace

auto detect_file_type(std::string_view path,
                      std::span<const std::byte> data) -> FileType {
    // --- Incompressible: already compressed formats ---
    // ZIP / JAR / APK / DOCX / XLSX / PPTX / ODT / EPUB
    static const std::uint8_t zip_magic[] = {0x50, 0x4B, 0x03, 0x04};
    if (starts_with(data, zip_magic, 4)) return FileType::Incompressible;

    // GZIP
    static const std::uint8_t gz_magic[] = {0x1F, 0x8B};
    if (starts_with(data, gz_magic, 2)) return FileType::Incompressible;

    // XZ
    static const std::uint8_t xz_magic[] = {0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00};
    if (starts_with(data, xz_magic, 6)) return FileType::Incompressible;

    // Zstd
    static const std::uint8_t zst_magic[] = {0x28, 0xB5, 0x2F, 0xFD};
    if (starts_with(data, zst_magic, 4)) return FileType::Incompressible;

    // BZIP2
    static const std::uint8_t bz2_magic[] = {0x42, 0x5A, 0x68};
    if (starts_with(data, bz2_magic, 3)) return FileType::Incompressible;

    // RAR
    static const std::uint8_t rar_magic[] = {0x52, 0x61, 0x72, 0x21, 0x1A, 0x07};
    if (starts_with(data, rar_magic, 6)) return FileType::Incompressible;

    // 7z
    static const std::uint8_t sz_magic[] = {0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C};
    if (starts_with(data, sz_magic, 6)) return FileType::Incompressible;

    // JPEG
    static const std::uint8_t jpg_magic[] = {0xFF, 0xD8, 0xFF};
    if (starts_with(data, jpg_magic, 3)) return FileType::Incompressible;

    // PNG
    static const std::uint8_t png_magic[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (starts_with(data, png_magic, 8)) return FileType::Incompressible;

    // GIF
    static const std::uint8_t gif_magic[] = {0x47, 0x49, 0x46, 0x38};
    if (starts_with(data, gif_magic, 4)) return FileType::Incompressible;

    // WebP (RIFF....WEBP)
    if (data.size() >= 12 &&
        std::memcmp(data.data(), "RIFF", 4) == 0 &&
        std::memcmp(data.data() + 8, "WEBP", 4) == 0) {
        return FileType::Incompressible;
    }

    // MP4 / MOV (ftyp box at offset 4)
    if (data.size() >= 8 && std::memcmp(data.data() + 4, "ftyp", 4) == 0) {
        return FileType::Incompressible;
    }

    // MP3
    static const std::uint8_t mp3_magic[] = {0xFF, 0xFB};
    if (starts_with(data, mp3_magic, 2)) return FileType::Incompressible;

    // OGG
    static const std::uint8_t ogg_magic[] = {0x4F, 0x67, 0x67, 0x53};
    if (starts_with(data, ogg_magic, 4)) return FileType::Incompressible;

    // FLAC
    static const std::uint8_t flac_magic[] = {0x66, 0x4C, 0x61, 0x43};
    if (starts_with(data, flac_magic, 4)) return FileType::Incompressible;

    // AVI
    if (data.size() >= 12 &&
        std::memcmp(data.data(), "RIFF", 4) == 0 &&
        std::memcmp(data.data() + 8, "AVI ", 4) == 0) {
        return FileType::Incompressible;
    }

    // Check extension-based incompressible types.
    auto ext = extension_of(path);
    static const char* incomp_exts[] = {
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".ico", ".tga",
        ".mp3", ".mp4", ".m4a", ".m4v", ".avi", ".mkv", ".mov", ".wmv",
        ".flv", ".ogg", ".opus", ".flac", ".wav", ".aac",
        ".zip", ".gz", ".xz", ".zst", ".bz2", ".rar", ".7z", ".tar",
        ".jar", ".apk", ".docx", ".xlsx", ".pptx", ".odt", ".epub",
        ".pdf", ".ps", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    };
    for (auto* e : incomp_exts) {
        if (ext == e) return FileType::Incompressible;
    }

    // --- Executables ---
    // PE (MZ header)
    static const std::uint8_t pe_magic[] = {0x4D, 0x5A};
    if (starts_with(data, pe_magic, 2)) return FileType::Executable;

    // ELF
    static const std::uint8_t elf_magic[] = {0x7F, 0x45, 0x4C, 0x46};
    if (starts_with(data, elf_magic, 4)) return FileType::Executable;

    // Mach-O (32-bit and 64-bit, big and little endian)
    if (data.size() >= 4) {
        auto m = *reinterpret_cast<const std::uint32_t*>(data.data());
        if (m == 0xFEEDFACEu || m == 0xFEEDFACFu ||
            m == 0xCEFAEDFEu || m == 0xCFFAEDFEu) {
            return FileType::Executable;
        }
    }

    auto ext_exec = extension_of(path);
    static const char* exec_exts[] = {
        ".exe", ".dll", ".sys", ".so", ".dylib", ".o", ".obj", ".lib", ".a",
    };
    for (auto* e : exec_exts) {
        if (ext_exec == e) return FileType::Executable;
    }

    // --- Text: source code, markup, data ---
    // Quick heuristic: if the first 512 bytes contain no null bytes, it's
    // likely text. Combined with extension check for source types.
    bool has_null = false;
    auto check_len = std::min(data.size(), std::size_t{512});
    for (std::size_t i = 0; i < check_len; ++i) {
        if (data[i] == std::byte{0}) { has_null = true; break; }
    }

    auto ext_text = extension_of(path);
    static const char* text_exts[] = {
        ".txt", ".text", ".log", ".csv", ".tsv", ".md", ".rst",
        ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hxx",
        ".cs", ".java", ".kt", ".scala", ".groovy",
        ".py", ".pyw", ".pyx", ".pxd",
        ".rb", ".pl", ".pm", ".php", ".lua", ".tcl",
        ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
        ".html", ".htm", ".xhtml", ".xml", ".svg", ".xsl",
        ".css", ".scss", ".sass", ".less",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties",
        ".sh", ".bash", ".zsh", ".fish", ".bat", ".cmd", ".ps1",
        ".sql", ".ddl", ".dml",
        ".r", ".R", ".jl", ".m", ".mm",
        ".go", ".rs", ".zig", ".nim", ".d", ".v", ".vhd", ".vhdl",
        ".hs", ".elm", ".ml", ".mli", ".ex", ".exs", ".erl", ".hrl",
        ".clj", ".cljs", ".lisp", ".el", ".scm", ".rkt",
        ".proto", ".graphql", ".gql", ".thrift",
        ".cmake", ".mk", ".makefile",
        ".dockerfile", ".dockerignore", ".gitignore", ".editorconfig",
        ".env", ".dotenv",
    };
    for (auto* e : text_exts) {
        if (ext_text == e) return FileType::Text;
    }

    // If no null bytes in the first 512 bytes, treat as text.
    if (!has_null) return FileType::Text;

    return FileType::Binary;
}

}  // namespace fzip
