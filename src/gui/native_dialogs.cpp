#include "native_dialogs.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
#include <shobjidl.h>

namespace fzip::gui {

namespace {

struct ComInit {
    HRESULT hr;
    ComInit() : hr(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)) {}
    ~ComInit() { if (SUCCEEDED(hr)) CoUninitialize(); }
};

std::string utf16_to_utf8(const wchar_t* wstr) {
    if (!wstr || !*wstr) return "";
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 1) return "";
    std::string str(len - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, str.data(), len, nullptr, nullptr);
    return str;
}

std::wstring utf8_to_utf16(const std::string& str) {
    if (str.empty()) return L"";
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    if (len <= 1) return L"";
    std::wstring wstr(len - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, wstr.data(), len);
    return wstr;
}

}  // namespace

auto pick_file(const std::string& title) -> std::string {
    ComInit com;
    IFileOpenDialog* pFileOpen = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL,
                                  IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));
    if (FAILED(hr)) return "";

    std::wstring wtitle = utf8_to_utf16(title);
    pFileOpen->SetTitle(wtitle.c_str());

    std::string result;
    hr = pFileOpen->Show(nullptr);
    if (SUCCEEDED(hr)) {
        IShellItem* pItem = nullptr;
        hr = pFileOpen->GetResult(&pItem);
        if (SUCCEEDED(hr)) {
            PWSTR pszFilePath = nullptr;
            hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);
            if (SUCCEEDED(hr)) {
                result = utf16_to_utf8(pszFilePath);
                CoTaskMemFree(pszFilePath);
            }
            pItem->Release();
        }
    }
    pFileOpen->Release();
    return result;
}

auto pick_files(const std::string& title) -> std::vector<std::string> {
    ComInit com;
    IFileOpenDialog* pFileOpen = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL,
                                  IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));
    if (FAILED(hr)) return {};

    std::wstring wtitle = utf8_to_utf16(title);
    pFileOpen->SetTitle(wtitle.c_str());
    FILEOPENDIALOGOPTIONS opt = 0;
    pFileOpen->GetOptions(&opt);
    pFileOpen->SetOptions(opt | FOS_ALLOWMULTISELECT);

    std::vector<std::string> results;
    hr = pFileOpen->Show(nullptr);
    if (SUCCEEDED(hr)) {
        IShellItemArray* pItems = nullptr;
        hr = pFileOpen->GetResults(&pItems);
        if (SUCCEEDED(hr)) {
            DWORD count = 0;
            pItems->GetCount(&count);
            for (DWORD i = 0; i < count; ++i) {
                IShellItem* pItem = nullptr;
                if (SUCCEEDED(pItems->GetItemAt(i, &pItem))) {
                    PWSTR pszFilePath = nullptr;
                    if (SUCCEEDED(pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath))) {
                        results.push_back(utf16_to_utf8(pszFilePath));
                        CoTaskMemFree(pszFilePath);
                    }
                    pItem->Release();
                }
            }
            pItems->Release();
        }
    }
    pFileOpen->Release();
    return results;
}

auto pick_folder(const std::string& title) -> std::string {
    ComInit com;
    IFileOpenDialog* pFileOpen = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL,
                                  IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));
    if (FAILED(hr)) return "";

    std::wstring wtitle = utf8_to_utf16(title);
    pFileOpen->SetTitle(wtitle.c_str());
    FILEOPENDIALOGOPTIONS opt = 0;
    pFileOpen->GetOptions(&opt);
    pFileOpen->SetOptions(opt | FOS_PICKFOLDERS);

    std::string result;
    hr = pFileOpen->Show(nullptr);
    if (SUCCEEDED(hr)) {
        IShellItem* pItem = nullptr;
        hr = pFileOpen->GetResult(&pItem);
        if (SUCCEEDED(hr)) {
            PWSTR pszFilePath = nullptr;
            hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);
            if (SUCCEEDED(hr)) {
                result = utf16_to_utf8(pszFilePath);
                CoTaskMemFree(pszFilePath);
            }
            pItem->Release();
        }
    }
    pFileOpen->Release();
    return result;
}

auto pick_save_file(const std::string& default_filename, const std::string& title) -> std::string {
    ComInit com;
    IFileSaveDialog* pFileSave = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_ALL,
                                  IID_IFileSaveDialog, reinterpret_cast<void**>(&pFileSave));
    if (FAILED(hr)) return "";

    std::wstring wtitle = utf8_to_utf16(title);
    pFileSave->SetTitle(wtitle.c_str());

    COMDLG_FILTERSPEC filter[] = {
        { L"ZIP Archives (*.zip)", L"*.zip" },
        { L"All Files (*.*)", L"*.*" }
    };
    pFileSave->SetFileTypes(2, filter);
    pFileSave->SetDefaultExtension(L"zip");

    std::wstring wdefault = utf8_to_utf16(default_filename);
    pFileSave->SetFileName(wdefault.c_str());

    std::string result;
    hr = pFileSave->Show(nullptr);
    if (SUCCEEDED(hr)) {
        IShellItem* pItem = nullptr;
        hr = pFileSave->GetResult(&pItem);
        if (SUCCEEDED(hr)) {
            PWSTR pszFilePath = nullptr;
            hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);
            if (SUCCEEDED(hr)) {
                result = utf16_to_utf8(pszFilePath);
                CoTaskMemFree(pszFilePath);
            }
            pItem->Release();
        }
    }
    pFileSave->Release();
    return result;
}

}  // namespace fzip::gui

#else

// Linux / macOS fallback using zenity or kdialog
#include <cstdio>
#include <array>
#include <memory>

namespace fzip::gui {

namespace {

std::string run_dialog_cmd(const std::string& cmd) {
    std::array<char, 512> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) return "";
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    // Trim trailing newline
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result;
}

}  // namespace

auto pick_file(const std::string& title) -> std::string {
    return run_dialog_cmd("zenity --file-selection --title=\"" + title + "\" 2>/dev/null");
}

auto pick_files(const std::string& title) -> std::vector<std::string> {
    std::string out = run_dialog_cmd("zenity --file-selection --multiple --separator=\"|\" --title=\"" + title + "\" 2>/dev/null");
    std::vector<std::string> res;
    size_t start = 0;
    while (start < out.size()) {
        size_t end = out.find('|', start);
        if (end == std::string::npos) {
            res.push_back(out.substr(start));
            break;
        }
        res.push_back(out.substr(start, end - start));
        start = end + 1;
    }
    return res;
}

auto pick_folder(const std::string& title) -> std::string {
    return run_dialog_cmd("zenity --file-selection --directory --title=\"" + title + "\" 2>/dev/null");
}

auto pick_save_file(const std::string& default_filename, const std::string& title) -> std::string {
    return run_dialog_cmd("zenity --file-selection --save --confirm-overwrite --filename=\"" + default_filename + "\" --title=\"" + title + "\" 2>/dev/null");
}

}  // namespace fzip::gui

#endif
