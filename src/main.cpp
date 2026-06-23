// fzip — state-of-the-art ZIP compressor in C++20.
// Entry point. Wires up the CLI subcommands stage by stage.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "codec.hpp"
#include "io.hpp"
#include "zip_writer.hpp"

namespace fzip {

extern const char kVersion[];

namespace {

void print_usage() {
    std::printf(
        "fzip %s — state-of-the-art ZIP compressor\n\n"
        "Usage:\n"
        "  fzip --version | -v\n"
        "  fzip --help    | -h\n"
        "  fzip store <archive.zip> <files...>\n"
        "  fzip deflate <archive.zip> <files...> [--level=N]   (Stage 3)\n"
        "  fzip zstd <archive.zip> <files...> [--level=N]      (Stage 4)\n"
        "  fzip auto <archive.zip> <files...>                  (Stage 5)\n"
        "\n"
        "A file argument beginning with '@' is treated as a list file: each\n"
        "line is read as one file path (e.g. '@inputs.txt').\n"
        "\n"
        "Stage 1 implements `store` (compression method 0).\n",
        kVersion);
}

auto parse_level(int argc, char** argv, int default_level) -> int {
    for (int i = 0; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a.starts_with("--level=")) {
            int v = 0;
            bool ok = !a.substr(8).empty();
            for (char c : a.substr(8)) {
                if (c < '0' || c > '9') { ok = false; break; }
                v = v * 10 + (c - '0');
            }
            if (ok && v >= 0) return v;
        }
    }
    return default_level;
}

// Expand any '@listfile' arguments into their contents. Each line of the
// list file is one path (blank lines and lines starting with '#' are
// skipped). Returns the expanded list of file paths.
auto expand_file_args(std::span<const std::string> args)
    -> std::vector<std::string> {
    std::vector<std::string> out;
    for (const auto& a : args) {
        if (!a.empty() && a[0] == '@') {
            std::ifstream lf(a.substr(1));
            if (!lf) {
                throw std::runtime_error("cannot open list file: " + a);
            }
            std::string line;
            while (std::getline(lf, line)) {
                if (!line.empty() && line[0] != '#') {
                    out.push_back(std::move(line));
                }
            }
        } else {
            out.push_back(a);
        }
    }
    return out;
}

// Read files from disk into ZipEntry structs (shared by all subcommands).
auto read_entries(const std::vector<std::string>& files)
    -> std::vector<ZipEntry> {
    namespace fs = std::filesystem;
    std::vector<ZipEntry> entries;
    entries.reserve(files.size());
    for (const auto& path : files) {
        fs::path p(path);
        ZipEntry e;
        e.name = p.filename().string();
        e.data = io::read_file(path);
        auto ftime = fs::last_write_time(p);
        auto sctime = std::chrono::time_point_cast<std::chrono::seconds>(ftime);
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(
            sctime.time_since_epoch());
        e.mtime_unix = static_cast<std::uint64_t>(secs.count());
        auto [dos_t, dos_d] = dos_time_date(e.mtime_unix);
        e.mod_time = dos_t;
        e.mod_date = dos_d;
        entries.push_back(std::move(e));
    }
    return entries;
}

auto cmd_compress(CodecId codec, int default_level, std::string_view mode_name,
                  int argc, char** argv) -> int {
    if (argc < 3) {
        std::fprintf(stderr, "fzip %.*s: need <archive.zip> <files...>\n",
                     static_cast<int>(mode_name.size()), mode_name.data());
        return 1;
    }
    std::string archive = argv[2];
    std::vector<std::string> raw_args;
    for (int i = 3; i < argc; ++i) {
        raw_args.emplace_back(argv[i]);
    }
    auto files = expand_file_args(raw_args);
    // Drop any command-line option flags (--level=..., etc.) from the file list.
    std::erase_if(files, [](const std::string& s) {
        return s.starts_with("--");
    });
    if (files.empty()) {
        std::fprintf(stderr, "fzip %.*s: no input files\n",
                     static_cast<int>(mode_name.size()), mode_name.data());
        return 1;
    }
    int level = parse_level(argc, argv, default_level);
    auto entries = read_entries(files);
    if (!write_zip(archive, entries, codec, level)) {
        std::fprintf(stderr, "fzip %.*s: failed to write '%s'\n",
                     static_cast<int>(mode_name.size()), mode_name.data(),
                     archive.c_str());
        return 1;
    }
    std::printf("fzip %.*s: wrote %s with %zu file(s)\n",
                static_cast<int>(mode_name.size()), mode_name.data(),
                archive.c_str(), files.size());
    return 0;
}

auto cmd_store(int argc, char** argv) -> int {
    return cmd_compress(CodecId::Store, 0, "store", argc, argv);
}

auto cmd_deflate(int argc, char** argv) -> int {
    return cmd_compress(CodecId::Deflate, 6, "deflate", argc, argv);
}

auto cmd_zstd(int argc, char** argv) -> int {
    return cmd_compress(CodecId::Zstd, 19, "zstd", argc, argv);
}

auto cmd_auto(int argc, char** argv) -> int {
    // Stage 5 will replace this with real auto-selection. For now, deflate.
    return cmd_compress(CodecId::Deflate, 6, "auto", argc, argv);
}

}  // namespace

auto run(int argc, char** argv) -> int {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    std::string_view cmd = argv[1];

    if (cmd == "--version" || cmd == "-v") {
        std::printf("fzip %s\n", kVersion);
        return 0;
    }
    if (cmd == "--help" || cmd == "-h") {
        print_usage();
        return 0;
    }
    if (cmd == "store") {
        return cmd_store(argc, argv);
    }
    if (cmd == "deflate") {
        return cmd_deflate(argc, argv);
    }
    if (cmd == "zstd") {
        return cmd_zstd(argc, argv);
    }
    if (cmd == "auto") {
        return cmd_auto(argc, argv);
    }

    std::fprintf(stderr, "fzip: unknown command '%.*s'. Try 'fzip --help'.\n",
                 static_cast<int>(cmd.size()), cmd.data());
    return 1;
}

}  // namespace fzip

const char fzip::kVersion[] = "0.1.0";

auto main(int argc, char** argv) -> int {
    try {
        return fzip::run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fzip: error: %s\n", e.what());
        return 2;
    }
}
