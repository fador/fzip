// fzip — state-of-the-art ZIP compressor in C++20.
// Entry point. Stage 0: print version and exit. Real CLI wiring per stage.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace fzip {

extern const char kVersion[];

auto run(int argc, char** argv) -> int {
    if (argc >= 2 && (std::string_view{argv[1]} == "--version" ||
                      std::string_view{argv[1]} == "-v")) {
        std::printf("fzip %s\n", kVersion);
        return 0;
    }
    if (argc >= 2 && (std::string_view{argv[1]} == "--help" ||
                      std::string_view{argv[1]} == "-h")) {
        std::printf(
            "fzip %s — state-of-the-art ZIP compressor\n\n"
            "Usage:\n"
            "  fzip --version\n"
            "  fzip --help\n"
            "\n"
            "Stages 1+ will add: fzip <archive.zip> <files...>\n",
            kVersion);
        return 0;
    }
    std::fprintf(stderr,
                 "fzip: no command given. Try 'fzip --help'.\n");
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
