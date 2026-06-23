// fzip — File-type detection implementation (Stage 5 stub).
#include "file_type.hpp"

namespace fzip {

auto detect_file_type(std::string_view /*path*/,
                      std::span<const std::byte> /*data*/) -> FileType {
    return FileType::Binary;  // Refined in Stage 5.
}

}  // namespace fzip
