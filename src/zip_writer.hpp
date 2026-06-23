// fzip — ZIP container writer.
// Stage 0 stub; Stage 1 implements local/central/EOCD headers for method 0.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fzip {

// Placeholder API to be filled in by Stage 1.
// Will write a ZIP archive with the given files using the Store method.
auto write_store_zip(const std::string& archive_path,
                     const std::vector<std::string>& file_paths) -> bool;

}  // namespace fzip
