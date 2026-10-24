#pragma once

#include <string>
#include <vector>

namespace fzip::gui {

// Open file picker dialog. Returns empty string if cancelled.
auto pick_file(const std::string& title = "Select File") -> std::string;

// Open multiple files picker dialog.
auto pick_files(const std::string& title = "Select Files") -> std::vector<std::string>;

// Open folder picker dialog. Returns empty string if cancelled.
auto pick_folder(const std::string& title = "Select Folder") -> std::string;

// Open save file dialog (e.g. for .zip). Returns empty string if cancelled.
auto pick_save_file(const std::string& default_filename = "archive.zip",
                    const std::string& title = "Save ZIP Archive") -> std::string;

}  // namespace fzip::gui
