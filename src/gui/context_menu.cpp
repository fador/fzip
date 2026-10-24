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

}  // namespace

auto check_context_menu_status() -> ContextMenuStatus {
    if (reg_key_exists(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.compress")) {
        return ContextMenuStatus::Registered;
    }
    return ContextMenuStatus::NotRegistered;
}

auto register_context_menu(const std::string& exe_path) -> bool {
    std::wstring wexe = to_wide(exe_path);
    std::wstring quote_exe = L"\"" + wexe + L"\"";

    bool ok = true;

    // 1. All files: "Compress with fzip..."
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.compress", L"", L"Compress with fzip...");
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.compress", L"Icon", quote_exe);
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.compress\\command", L"", quote_exe + L" --compress-dialog \"%1\"");

    // Quick compress to .zip
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.quick", L"", L"Quick Compress to .zip (fzip)");
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.quick", L"Icon", quote_exe);
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.quick\\command", L"", quote_exe + L" --quick-compress \"%1\"");

    // 2. Directories: "Compress with fzip..."
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\Directory\\shell\\fzip.compress", L"", L"Compress with fzip...");
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\Directory\\shell\\fzip.compress", L"Icon", quote_exe);
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\Directory\\shell\\fzip.compress\\command", L"", quote_exe + L" --compress-dialog \"%1\"");

    // Quick compress directory to .zip
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\Directory\\shell\\fzip.quick", L"", L"Quick Compress to .zip (fzip)");
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\Directory\\shell\\fzip.quick", L"Icon", quote_exe);
    ok &= reg_set_string(HKEY_CURRENT_USER, L"Software\\Classes\\Directory\\shell\\fzip.quick\\command", L"", quote_exe + L" --quick-compress \"%1\"");

    // 3. .zip archives: "Extract Here (fzip)" and "Extract to folder (fzip)"
    std::wstring zip_base = L"Software\\Classes\\SystemFileAssociations\\.zip\\shell";
    ok &= reg_set_string(HKEY_CURRENT_USER, zip_base + L"\\fzip.extract_here", L"", L"Extract Here (fzip)");
    ok &= reg_set_string(HKEY_CURRENT_USER, zip_base + L"\\fzip.extract_here", L"Icon", quote_exe);
    ok &= reg_set_string(HKEY_CURRENT_USER, zip_base + L"\\fzip.extract_here\\command", L"", quote_exe + L" --extract-here \"%1\"");

    ok &= reg_set_string(HKEY_CURRENT_USER, zip_base + L"\\fzip.extract_sub", L"", L"Extract to folder (fzip)");
    ok &= reg_set_string(HKEY_CURRENT_USER, zip_base + L"\\fzip.extract_sub", L"Icon", quote_exe);
    ok &= reg_set_string(HKEY_CURRENT_USER, zip_base + L"\\fzip.extract_sub\\command", L"", quote_exe + L" --extract-subdir \"%1\"");

    ok &= reg_set_string(HKEY_CURRENT_USER, zip_base + L"\\fzip.inspect", L"", L"Open with fzip");
    ok &= reg_set_string(HKEY_CURRENT_USER, zip_base + L"\\fzip.inspect", L"Icon", quote_exe);
    ok &= reg_set_string(HKEY_CURRENT_USER, zip_base + L"\\fzip.inspect\\command", L"", quote_exe + L" \"%1\"");

    return ok;
}

auto unregister_context_menu() -> bool {
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.compress");
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\*\\shell\\fzip.quick");
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\Directory\\shell\\fzip.compress");
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\Directory\\shell\\fzip.quick");
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\SystemFileAssociations\\.zip\\shell\\fzip.extract_here");
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\SystemFileAssociations\\.zip\\shell\\fzip.extract_sub");
    reg_delete_tree(HKEY_CURRENT_USER, L"Software\\Classes\\SystemFileAssociations\\.zip\\shell\\fzip.inspect");
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
