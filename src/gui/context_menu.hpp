#pragma once

#include <string>

namespace fzip::gui {

enum class ContextMenuStatus {
    Registered,
    NotRegistered,
    Unknown
};

// Check if context menu items are currently registered in user's registry / config.
auto check_context_menu_status() -> ContextMenuStatus;

// Register context menus for the current user (no admin rights needed).
// Windows: writes to HKCU\Software\Classes
// Linux: writes to ~/.local/share/kio/servicemenus and ~/.local/share/nautilus/scripts
// Returns true on success.
auto register_context_menu(const std::string& exe_path) -> bool;

// Unregister context menus.
auto unregister_context_menu() -> bool;

}  // namespace fzip::gui
