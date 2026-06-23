// fzip — state-of-the-art ZIP compressor in C++20.
// Entry point. Wires up the CLI subcommands stage by stage.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "codec.hpp"
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

auto cmd_store(int argc, char** argv) -> int {
    // argv[0]=store argv[1]=archive argv[2..]=files / @listfiles
    if (argc < 3) {
        std::fprintf(stderr, "fzip store: need <archive.zip> <files...>\n");
        return 1;
    }
    std::string archive = argv[2];
    std::vector<std::string> raw_args;
    for (int i = 3; i < argc; ++i) {
        raw_args.emplace_back(argv[i]);
    }
    auto files = expand_file_args(raw_args);
    if (files.empty()) {
        std::fprintf(stderr, "fzip store: no input files\n");
        return 1;
    }
    if (!write_store_zip(archive, files)) {
        std::fprintf(stderr, "fzip store: failed to write '%s'\n", archive.c_str());
        return 1;
    }
    std::printf("fzip store: wrote %s with %zu file(s)\n",
                archive.c_str(), files.size());
    return 0;
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
