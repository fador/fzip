// fzip — ZIP container writer implementation.
// Stage 0 stub. Stage 1 fills in real Store-method writing.
#include "zip_writer.hpp"

namespace fzip {

auto write_store_zip(const std::string& /*archive_path*/,
                     const std::vector<std::string>& /*file_paths*/) -> bool {
    return false;  // Not implemented until Stage 1.
}

}  // namespace fzip
