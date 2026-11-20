#include "context_menu.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace fzip::gui {

#if defined(_WIN32)

namespace {

bool reg_set_string(HKEY root, const std::wstring& subkey, const std::wstring& name, const std::wstring& value) {
    HKEY hkey = nullptr;
    LSTATUS status = RegCreateKeyExW(root, subkey.c_str(), 0, nullptr,
                                     REG_OPTION_NON_VOLATILE, KEY_SET_VALUE,
                                     nullptr, &hkey, nullptr);
    if (status != ERROR_SUCCESS) return false;
    status = RegSetValueExW(hkey, name.empty() ? nullptr : name.c_str(), 0, REG_SZ,
                            reinterpret_cast<const BYTE*>(value.c_str()),
                            static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(hkey);
    return status == ERROR_SUCCESS;
}

bool reg_key_exists(HKEY root, const std::wstring& subkey) {
    HKEY hkey = nullptr;
    LSTATUS status = RegOpenKeyExW(root, subkey.c_str(), 0, KEY_READ, &hkey);
    if (status == ERROR_SUCCESS) {
        RegCloseKey(hkey);
        return true;
    }
    return false;
}

void reg_delete_tree(HKEY root, const std::wstring& subkey) {
    RegDeleteTreeW(root, subkey.c_str());
}

std::wstring to_wide(const std::string& str) {
    if (str.empty()) return L"";
    int count = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
    std::wstring w(count, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), w.data(), count);
    return w;
}

bool register_verb(const std::wstring& shell_root,
                   const std::wstring& key_name,
                   const std::wstring& label,
                   const std::wstring& icon,
                   const std::wstring& command,
                   const wchar_t* multi_select_model = nullptr) {
    const std::wstring base = shell_root + L"\\" + key_name;
    bool ok = true;
    ok &= reg_set_string(HKEY_CURRENT_USER, base, L"", label);
    ok &= reg_set_string(HKEY_CURRENT_USER, base, L"Icon", icon);
    if (multi_select_model != nullptr) {
        ok &= reg_set_string(HKEY_CURRENT_USER, base, L"MultiSelectModel", multi_select_model);
    }
    ok &= reg_set_string(HKEY_CURRENT_USER, base + L"\\command", L"", command);
    return ok;
}

constexpr const wchar_t* kZipShellRoots[] = {
    L"Software\\Classes\\SystemFileAssociations\\.zip\\shell",
    L"Software\\Classes\\.zip\\shell",
    L"Software\\Classes\\CompressedFolder\\shell",
};

}  // namespace

auto check_context_menu_status() -> ContextMenuStatus {
    if (reg_key_exists(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.compress") &&
        reg_key_exists(HKEY_CURRENT_USER,
                       L"Software\\Classes\\SystemFileAssociations\\.zip\\shell\\fzip.extract_sub")) {
        return ContextMenuStatus::Registered;
    }
    return ContextMenuStatus::NotRegistered;
}

auto register_context_menu(const std::string& exe_path) -> bool {
    std::wstring wexe = to_wide(exe_path);
    std::wstring quote_exe = L"\"" + wexe + L"\"";
    std::wstring icon_app = quote_exe + L",0";
    std::wstring icon_compress = quote_exe + L",1";
    std::wstring icon_extract = quote_exe + L",2";

    bool ok = true;

    // 1. Files and directories: "Compress with fzip..." & "Quick Compress"
    for (const wchar_t* root : {
             L"Software\\Classes\\*\\shell",
             L"Software\\Classes\\Directory\\shell",
         }) {
        ok &= register_verb(root, L"fzip.compress", L"Compress with fzip...",
                            icon_compress, quote_exe + L" --compress-dialog \"%1\"", L"Player");
        ok &= register_verb(root, L"fzip.quick", L"Quick Compress to .zip (fzip)",
                            icon_compress, quote_exe + L" --quick-compress \"%1\"", L"Player");
    }

    // 2. .zip archives across all Explorer association lookup paths
    for (const wchar_t* zip_base : kZipShellRoots) {
        ok &= register_verb(zip_base, L"fzip.extract_here", L"Extract Here (fzip)",
                            icon_extract, quote_exe + L" --extract-here \"%1\"");
        ok &= register_verb(zip_base, L"fzip.extract_sub", L"Extract to folder... (fzip)",
                            icon_extract, quote_exe + L" --extract-subdir \"%1\"");
        ok &= register_verb(zip_base, L"fzip.inspect", L"Open with fzip",
                            icon_app, quote_exe + L" \"%1\"");
    }

    return ok;
}

auto unregister_context_menu() -> bool {
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.compress");
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.quick");
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\Directory\\shell\\fzip.compress");
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\Directory\\shell\\fzip.quick");
    for (const wchar_t* zip_base : kZipShellRoots) {
        const std::wstring base(zip_base);
        reg_delete_tree(HKEY_CURRENT_USER, base + L"\\fzip.extract_here");
        reg_delete_tree(HKEY_CURRENT_USER, base + L"\\fzip.extract_sub");
        reg_delete_tree(HKEY_CURRENT_USER, base + L"\\fzip.inspect");
    }
    return true;
}

#else

// Linux / macOS fallback using FreeDesktop / KDE service menus & Nautilus scripts
namespace fs = std::filesystem;

static fs::path get_home_dir() {
    const char* home = std::getenv("HOME");
    return home ? fs::path(home) : fs::path();
}

auto check_context_menu_status() -> ContextMenuStatus {
    fs::path home = get_home_dir();
    if (home.empty()) return ContextMenuStatus::Unknown;
    fs::path kio = home / ".local/share/kio/servicemenus/fzip.desktop";
    if (fs::exists(kio)) return ContextMenuStatus::Registered;
    return ContextMenuStatus::NotRegistered;
}

auto register_context_menu(const std::string& exe_path) -> bool {
    fs::path home = get_home_dir();
    if (home.empty()) return false;

    std::error_code ec;
    // 1. KDE ServiceMenu
    fs::path kio_dir = home / ".local/share/kio/servicemenus";
    fs::create_directories(kio_dir, ec);
    fs::path kio_file = kio_dir / "fzip.desktop";
    std::ofstream kout(kio_file);
    if (kout) {
        kout << "[Desktop Entry]\n"
             << "Type=Service\n"
             << "ServiceTypes=KonqPopupMenu/Plugin\n"
             << "MimeType=all/allfiles;inode/directory;\n"
             << "Actions=fzipCompress;fzipQuick;\n"
             << "X-KDE-Submenu=fzip\n\n"
             << "[Desktop Action fzipCompress]\n"
             << "Name=Compress with fzip...\n"
             << "Exec=\"" << exe_path << "\" --compress-dialog %F\n\n"
             << "[Desktop Action fzipQuick]\n"
             << "Name=Quick Compress to .zip (fzip)\n"
             << "Exec=\"" << exe_path << "\" --quick-compress %F\n";
    }

    // 2. GNOME Nautilus / Nemo script
    fs::path nautilus_dir = home / ".local/share/nautilus/scripts";
    fs::create_directories(nautilus_dir, ec);
    fs::path nscript = nautilus_dir / "Compress with fzip";
    std::ofstream nout(nscript);
    if (nout) {
        nout << "#!/bin/sh\n"
             << "\"" << exe_path << "\" --compress-dialog \"$@\"\n";
    }
    fs::permissions(nscript, fs::perms::owner_all, fs::perm_options::add, ec);

    return true;
}

auto unregister_context_menu() -> bool {
    fs::path home = get_home_dir();
    if (home.empty()) return false;
    std::error_code ec;
    fs::remove(home / ".local/share/kio/servicemenus/fzip.desktop", ec);
    fs::remove(home / ".local/share/nautilus/scripts/Compress with fzip", ec);
    return true;
}

#endif

}  // namespace fzip::gui
