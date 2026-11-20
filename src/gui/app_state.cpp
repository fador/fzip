#include "app_state.hpp"
#include "native_dialogs.hpp"
#include "zip_reader.hpp"
#include "zip_writer.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace fzip::gui {

namespace fs = std::filesystem;

namespace {

auto format_size(std::uint64_t bytes) -> std::string {
    constexpr double kKiB = 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    if (bytes >= kGiB) {
        oss << (bytes / kGiB) << " GiB";
    } else if (bytes >= kMiB) {
        oss << (bytes / kMiB) << " MiB";
    } else if (bytes >= kKiB) {
        oss << (bytes / kKiB) << " KiB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

auto codec_name(CodecId id) -> const char* {
    switch (id) {
        case CodecId::Store: return "Store";
        case CodecId::Deflate: return "Deflate";
        case CodecId::Zstd: return "Zstd";
        case CodecId::Fzac: return "FZAC";
    }
    return "Unknown";
}

auto codec_color(CodecId id) -> ImVec4 {
    switch (id) {
        case CodecId::Fzac: return ImVec4(0.74f, 0.55f, 1.00f, 1.00f);
        case CodecId::Zstd: return ImVec4(0.35f, 0.65f, 1.00f, 1.00f);
        case CodecId::Deflate: return ImVec4(0.25f, 0.80f, 0.52f, 1.00f);
        case CodecId::Store: return ImVec4(0.55f, 0.60f, 0.68f, 1.00f);
    }
    return ImVec4(0.75f, 0.75f, 0.75f, 1.00f);
}

auto iequals(std::string_view a, std::string_view b) -> bool {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

auto icontains(std::string_view hay, std::string_view needle) -> bool {
    if (needle.empty()) return true;
    if (needle.size() > hay.size()) return false;
    for (std::size_t i = 0; i <= hay.size() - needle.size(); ++i) {
        if (iequals(hay.substr(i, needle.size()), needle)) return true;
    }
    return false;
}

void open_folder_in_explorer(const std::string& dir) {
#if defined(_WIN32)
    if (!dir.empty()) {
        ::ShellExecuteA(nullptr, "open", dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }
#else
    (void)dir;
#endif
}

auto styled_button(const char* label, const ImVec2& size,
                   const ImVec4& base, const ImVec4& hov, const ImVec4& act) -> bool {
    ImGui::PushStyleColor(ImGuiCol_Button, base);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hov);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, act);
    const bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(3);
    return clicked;
}

void draw_brand_badge(float size, bool extract_accent = false) {
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(size, size));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p1(p0.x + size, p0.y + size);
    const float r = size * 0.22f;
    const ImU32 bg_col = extract_accent ? IM_COL32(14, 34, 32, 255) : IM_COL32(16, 24, 38, 255);
    const ImU32 bd_col = extract_accent ? IM_COL32(32, 201, 151, 255) : IM_COL32(88, 166, 255, 255);
    const ImU32 zip_col = extract_accent ? IM_COL32(16, 185, 129, 255) : IM_COL32(188, 140, 255, 255);

    dl->AddRectFilled(p0, p1, bg_col, r);
    dl->AddRect(p0, p1, bd_col, r, 0, std::max(1.5f, size * 0.08f));

    // Zipper track + teeth
    const float zx = p0.x + size * 0.34f;
    dl->AddLine(ImVec2(zx, p0.y + size * 0.18f), ImVec2(zx, p0.y + size * 0.82f),
                IM_COL32(95, 118, 150, 255), std::max(1.5f, size * 0.07f));
    for (int k = 0; k < 3; ++k) {
        const float ty = p0.y + size * (0.24f + 0.14f * static_cast<float>(k));
        const float ox = (k % 2 == 0) ? -size * 0.14f : size * 0.02f;
        dl->AddRectFilled(ImVec2(zx + ox, ty),
                          ImVec2(zx + ox + size * 0.12f, ty + size * 0.07f),
                          IM_COL32(205, 222, 242, 255), 1.0f);
    }
    // Zipper slider pull tab
    dl->AddRectFilled(ImVec2(zx - size * 0.11f, p0.y + size * 0.62f),
                      ImVec2(zx + size * 0.11f, p0.y + size * 0.82f),
                      zip_col, 2.0f);

    // Accent bar on right side
    dl->AddRectFilled(ImVec2(p0.x + size * 0.60f, p0.y + size * 0.30f),
                      ImVec2(p0.x + size * 0.80f, p0.y + size * 0.42f),
                      bd_col, 1.5f);
    dl->AddRectFilled(ImVec2(p0.x + size * 0.60f, p0.y + size * 0.50f),
                      ImVec2(p0.x + size * 0.74f, p0.y + size * 0.62f),
                      zip_col, 1.5f);
}

void render_metric_card(const char* label, const std::string& value, const ImVec4& val_color) {
    ImGui::TextColored(ImVec4(0.56f, 0.62f, 0.71f, 1.00f), "%s", label);
    ImGui::TextColored(val_color, "%s", value.c_str());
}

auto get_exe_path() -> std::string {
    char exe_buf[1024] = {0};
#if defined(_WIN32)
    GetModuleFileNameA(nullptr, exe_buf, sizeof(exe_buf));
#else
    std::error_code ec;
    std::string p = fs::canonical("/proc/self/exe", ec).string();
    std::snprintf(exe_buf, sizeof(exe_buf), "%s", p.c_str());
#endif
    return std::string(exe_buf);
}

}  // namespace

AppState::AppState() {
    ctx_status_ = check_context_menu_status();
}

AppState::~AppState() {
    stop_ipc_server();
    cancel_operation();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

auto AppState::try_join_or_start_ipc(const char* pipe_id,
                                     const std::vector<std::string>& paths) -> bool {
#if defined(_WIN32)
    std::wstring wpipe = L"\\\\.\\pipe\\";
    for (const char* p = pipe_id; *p != '\0'; ++p) {
        wpipe.push_back(static_cast<wchar_t>(static_cast<unsigned char>(*p)));
    }

    HANDLE hPipe = ::CreateNamedPipeW(
        wpipe.c_str(),
        PIPE_ACCESS_INBOUND | FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        1, 0, 16384, 0, nullptr);

    if (hPipe != INVALID_HANDLE_VALUE) {
        HANDLE hStop = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
        ipc_pipe_handle_ = hPipe;
        ipc_stop_event_ = hStop;
        {
            std::lock_guard<std::mutex> lock(ipc_mutex_);
            last_ipc_time_ = std::chrono::steady_clock::now();
        }
        ipc_thread_ = std::thread([this, hPipe, hStop]() {
            HANDLE hConnEv = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
            while (::WaitForSingleObject(hStop, 0) != WAIT_OBJECT_0) {
                ::ResetEvent(hConnEv);
                OVERLAPPED ov{};
                ov.hEvent = hConnEv;
                BOOL ok = ::ConnectNamedPipe(hPipe, &ov);
                if (!ok) {
                    const DWORD err = ::GetLastError();
                    if (err == ERROR_IO_PENDING) {
                        HANDLE waits[2] = { hStop, hConnEv };
                        const DWORD w = ::WaitForMultipleObjects(2, waits, FALSE, INFINITE);
                        if (w != WAIT_OBJECT_0 + 1) {
                            ::CancelIoEx(hPipe, &ov);
                            break;
                        }
                    } else if (err != ERROR_PIPE_CONNECTED) {
                        if (::WaitForSingleObject(hStop, 10) == WAIT_OBJECT_0) break;
                        continue;
                    }
                }

                std::string payload;
                char buf[4096];
                for (;;) {
                    ::ResetEvent(hConnEv);
                    OVERLAPPED rov{};
                    rov.hEvent = hConnEv;
                    DWORD nread = 0;
                    BOOL rok = ::ReadFile(hPipe, buf, sizeof(buf), &nread, &rov);
                    if (!rok) {
                        const DWORD rerr = ::GetLastError();
                        if (rerr == ERROR_IO_PENDING) {
                            HANDLE waits[2] = { hStop, hConnEv };
                            const DWORD rw = ::WaitForMultipleObjects(2, waits, FALSE, INFINITE);
                            if (rw != WAIT_OBJECT_0 + 1) {
                                ::CancelIoEx(hPipe, &rov);
                                break;
                            }
                            if (!::GetOverlappedResult(hPipe, &rov, &nread, FALSE)) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    if (nread == 0) break;
                    payload.append(buf, nread);
                }

                std::vector<std::string> incoming;
                std::size_t start = 0;
                while (start < payload.size()) {
                    const std::size_t nl = payload.find('\n', start);
                    const std::size_t end = (nl == std::string::npos) ? payload.size() : nl;
                    if (end > start) {
                        incoming.push_back(payload.substr(start, end - start));
                    }
                    if (nl == std::string::npos) break;
                    start = nl + 1;
                }
                if (!incoming.empty()) {
                    std::lock_guard<std::mutex> lock(ipc_mutex_);
                    pending_ipc_paths_.insert(
                        pending_ipc_paths_.end(),
                        std::make_move_iterator(incoming.begin()),
                        std::make_move_iterator(incoming.end()));
                    last_ipc_time_ = std::chrono::steady_clock::now();
                }
                ::DisconnectNamedPipe(hPipe);
            }
            ::CloseHandle(hConnEv);
        });
        return false;
    }

    std::string payload;
    for (const auto& p : paths) {
        if (!p.empty()) {
            payload += p;
            payload.push_back('\n');
        }
    }
    for (int attempt = 0; attempt < 30; ++attempt) {
        HANDLE hClient = ::CreateFileW(
            wpipe.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (hClient != INVALID_HANDLE_VALUE) {
            std::size_t off = 0;
            bool write_ok = true;
            while (off < payload.size()) {
                DWORD written = 0;
                const DWORD chunk = static_cast<DWORD>(
                    std::min<std::size_t>(payload.size() - off, 65536u));
                if (!::WriteFile(hClient, payload.data() + off, chunk, &written, nullptr)) {
                    write_ok = false;
                    break;
                }
                off += written;
            }
            ::FlushFileBuffers(hClient);
            ::CloseHandle(hClient);
            if (write_ok) return true;
        }
        ::WaitNamedPipeW(wpipe.c_str(), 150);
    }
    return false;
#else
    (void)pipe_id;
    (void)paths;
    return false;
#endif
}

void AppState::stop_ipc_server() {
#if defined(_WIN32)
    if (ipc_stop_event_ != nullptr) {
        ::SetEvent(static_cast<HANDLE>(ipc_stop_event_));
    }
    if (ipc_thread_.joinable()) {
        ipc_thread_.join();
    }
    if (ipc_pipe_handle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(ipc_pipe_handle_));
        ipc_pipe_handle_ = nullptr;
    }
    if (ipc_stop_event_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(ipc_stop_event_));
        ipc_stop_event_ = nullptr;
    }
#endif
}

void AppState::drain_ipc_paths() {
    std::vector<std::string> incoming;
    {
        std::lock_guard<std::mutex> lock(ipc_mutex_);
        if (!pending_ipc_paths_.empty()) {
            incoming.swap(pending_ipc_paths_);
        }
    }
    if (!incoming.empty()) {
        add_input_paths(incoming);
#if defined(_WIN32)
        if (mode_ == AppMode::Compress && native_hwnd_ != nullptr) {
            HWND hwnd = static_cast<HWND>(native_hwnd_);
            if (::IsIconic(hwnd)) {
                ::ShowWindow(hwnd, SW_RESTORE);
            }
            ::SetForegroundWindow(hwnd);
        }
#endif
    }
}

void AppState::set_status(std::string msg) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_message_ = std::move(msg);
}

auto AppState::get_status() -> std::string {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_message_;
}

void AppState::init(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    if (args.empty()) {
        mode_ = AppMode::Compress;
        try_join_or_start_ipc("fzip_gui_compress_dialog", {});
        return;
    }

    std::string_view first = args[0];

    if (first == "--register-context-menu") {
        register_context_menu(get_exe_path());
        close_requested_ = true;
        return;
    }

    if (first == "--unregister-context-menu") {
        unregister_context_menu();
        close_requested_ = true;
        return;
    }

    if (first == "--quick-compress") {
        std::vector<std::string> input_paths(args.begin() + 1, args.end());
        if (try_join_or_start_ipc("fzip_gui_quick_compress", input_paths)) {
            close_requested_ = true;
            return;
        }
        mode_ = AppMode::QuickProgress;
        is_extract_op_ = false;
        add_input_paths(input_paths);
        pending_quick_start_ = true;
        set_status("Collecting selected files...");
        return;
    }

    if (first == "--compress-dialog") {
        std::vector<std::string> input_paths(args.begin() + 1, args.end());
        if (try_join_or_start_ipc("fzip_gui_compress_dialog", input_paths)) {
            close_requested_ = true;
            return;
        }
        mode_ = AppMode::Compress;
        add_input_paths(input_paths);
        return;
    }

    if (first == "--extract-here" && args.size() >= 2) {
        mode_ = AppMode::QuickProgress;
        is_extract_op_ = true;
        std::error_code ec;
        fs::path zip_path = fs::absolute(fs::path(args[1]), ec);
        if (zip_path.empty()) zip_path = fs::path(args[1]);
        inspected_archive_path_ = zip_path.string();
        extract_archive(zip_path.parent_path().string());
        return;
    }

    if (first == "--extract-subdir" && args.size() >= 2) {
        mode_ = AppMode::QuickProgress;
        is_extract_op_ = true;
        std::error_code ec;
        fs::path zip_path = fs::absolute(fs::path(args[1]), ec);
        if (zip_path.empty()) zip_path = fs::path(args[1]);
        inspected_archive_path_ = zip_path.string();
        fs::path out_dir = zip_path.parent_path() / zip_path.stem();
        extract_archive(out_dir.string());
        return;
    }

    if (args.size() == 1) {
        fs::path p(args[0]);
        if (iequals(p.extension().string(), ".zip")) {
            std::error_code ec;
            fs::path abs_p = fs::absolute(p, ec);
            open_archive(abs_p.empty() ? p.string() : abs_p.string());
            return;
        }
    }

    mode_ = AppMode::Compress;
    add_input_paths(args);
}

void AppState::add_input_paths(const std::vector<std::string>& paths) {
    if (paths.empty() || is_working_.load()) return;
    for (const auto& p_str : paths) {
        fs::path p(p_str);
        std::error_code ec;
        if (!fs::exists(p, ec)) continue;

        const std::string abs_path = fs::absolute(p, ec).string();
        const bool already_present = std::any_of(
            input_items_.begin(), input_items_.end(),
            [&](const FileItem& it) { return it.path == abs_path; });
        if (already_present) continue;

        FileItem item;
        item.path = abs_path;
        item.is_directory = fs::is_directory(p, ec);
        if (!item.is_directory) {
            item.size = fs::file_size(p, ec);
        } else {
            std::uint64_t dsize = 0;
            for (const auto& sub : fs::recursive_directory_iterator(p, ec)) {
                if (sub.is_regular_file(ec)) {
                    dsize += sub.file_size(ec);
                }
            }
            item.size = dsize;
        }
        input_items_.push_back(std::move(item));
    }

    if (auto_output_path_ && !input_items_.empty()) {
        fs::path first(input_items_[0].path);
        fs::path parent = first.parent_path();
        std::string archive_stem;
        if (input_items_.size() == 1) {
            archive_stem = first.stem().string();
        } else {
            archive_stem = parent.filename().string();
            if (archive_stem.empty()) archive_stem = "archive";
        }
        output_archive_path_ = (parent / (archive_stem + ".zip")).string();
    }
}

void AppState::open_archive(const std::string& path) {
    if (path.empty()) return;
    std::error_code ec;
    fs::path abs_p = fs::absolute(fs::path(path), ec);
    const std::string resolved = abs_p.empty() ? path : abs_p.string();

    try {
        auto entries = read_zip_entries(resolved);
        inspected_archive_path_ = resolved;
        inspected_entries_ = std::move(entries);
        inspected_total_uncomp_ = 0;
        inspected_total_comp_ = 0;
        for (const auto& e : inspected_entries_) {
            inspected_total_uncomp_ += e.uncompressed_size;
            inspected_total_comp_ += e.compressed_size;
        }
        last_extracted_dir_.clear();
        operation_failed_ = false;
        set_status("Loaded archive (" + std::to_string(inspected_entries_.size()) + " entries).");
        mode_ = AppMode::Inspect;
    } catch (const std::exception& e) {
        operation_failed_ = true;
        set_status(std::string("Failed to open archive: ") + e.what());
    }
}

void AppState::extract_archive(const std::string& out_dir) {
    if (is_working_.load()) return;

    const std::string archive_path =
        !inspected_archive_path_.empty() ? inspected_archive_path_ : output_archive_path_;
    if (archive_path.empty()) {
        operation_failed_ = true;
        set_status("Extraction error: no archive specified.");
        return;
    }

    std::error_code ec;
    fs::path arch_fs(archive_path);
    fs::path out_p = out_dir.empty()
        ? fs::absolute(arch_fs, ec).parent_path()
        : fs::absolute(fs::path(out_dir), ec);
    if (out_p.empty()) {
        out_p = fs::current_path(ec);
    }
    const std::string abs_out_dir = out_p.string();

    is_working_ = true;
    is_extract_op_ = true;
    operation_finished_ = false;
    operation_failed_ = false;
    cancel_requested_ = false;
    completed_count_ = 0;
    total_count_ = 0;
    total_uncompressed_bytes_ = 0;
    total_compressed_bytes_ = 0;
    elapsed_seconds_ = 0.0;
    mb_per_sec_ = 0.0;
    last_extracted_dir_ = abs_out_dir;
    start_time_ = std::chrono::steady_clock::now();
    set_status("Preparing extraction to " + abs_out_dir + "...");

    {
        std::lock_guard<std::mutex> lock(rows_mutex_);
        rows_.clear();
    }

    if (worker_thread_.joinable()) worker_thread_.join();

    worker_thread_ = std::thread([this, archive_path, out_p, abs_out_dir]() {
        try {
            if (!fs::exists(archive_path)) {
                throw std::runtime_error("archive not found: " + archive_path);
            }
            fs::create_directories(out_p);

            auto cd_entries = read_zip_entries(archive_path);
            total_count_ = cd_entries.size();

            std::uint64_t comp_sum = 0;
            {
                std::lock_guard<std::mutex> lock(rows_mutex_);
                rows_.resize(cd_entries.size());
                for (std::size_t i = 0; i < cd_entries.size(); ++i) {
                    rows_[i].name = cd_entries[i].name;
                    rows_[i].uncompressed_size = cd_entries[i].uncompressed_size;
                    rows_[i].compressed_size = cd_entries[i].compressed_size;
                    rows_[i].codec = cd_entries[i].codec;
                    comp_sum += cd_entries[i].compressed_size;
                }
            }
            total_compressed_bytes_ = comp_sum;
            set_status("Decompressing " + std::to_string(cd_entries.size()) + " entries...");

            auto entries = extract_all(archive_path);
            total_count_ = entries.size();
            std::size_t files_written = 0;

            for (std::size_t i = 0; i < entries.size(); ++i) {
                if (cancel_requested_.load()) break;
                const auto& [name, data] = entries[i];

                fs::path rel = fs::path(name).relative_path().lexically_normal();
                for (const auto& part : rel) {
                    if (part == "..") {
                        throw std::runtime_error("unsafe archive entry path: " + name);
                    }
                }
                if (rel.empty() || rel == ".") continue;

                const fs::path dest = out_p / rel;
                const bool is_dir_entry =
                    !name.empty() && (name.back() == '/' || name.back() == '\\');
                if (is_dir_entry) {
                    fs::create_directories(dest);
                } else {
                    if (dest.has_parent_path()) {
                        fs::create_directories(dest.parent_path());
                    }
                    std::ofstream out(dest, std::ios::binary);
                    if (!out) {
                        throw std::runtime_error("cannot create file: " + dest.string());
                    }
                    if (!data.empty()) {
                        out.write(reinterpret_cast<const char*>(data.data()),
                                  static_cast<std::streamsize>(data.size()));
                        if (!out) {
                            throw std::runtime_error("failed writing file: " + dest.string());
                        }
                    }
                    out.flush();
                    if (!out) {
                        throw std::runtime_error("failed flushing file: " + dest.string());
                    }
                    ++files_written;
                }

                total_uncompressed_bytes_ += data.size();
                {
                    std::lock_guard<std::mutex> lock(rows_mutex_);
                    if (i < rows_.size()) {
                        rows_[i].completed = true;
                    }
                }
                completed_count_ = i + 1;
            }

            finish_time_ = std::chrono::steady_clock::now();
            const double dt = std::max(
                0.001, std::chrono::duration<double>(finish_time_ - start_time_).count());
            elapsed_seconds_ = dt;
            mb_per_sec_ =
                (static_cast<double>(total_uncompressed_bytes_.load()) / (1024.0 * 1024.0)) / dt;

            if (cancel_requested_.load()) {
                set_status("Extraction cancelled by user.");
            } else {
                operation_finished_ = true;
                set_status("Extracted " + std::to_string(files_written) +
                           " files to " + abs_out_dir);
            }
        } catch (const std::exception& e) {
            operation_failed_ = true;
            finish_time_ = std::chrono::steady_clock::now();
            elapsed_seconds_ =
                std::chrono::duration<double>(finish_time_ - start_time_).count();
            set_status(std::string("Extraction error: ") + e.what());
        }
        is_working_ = false;
    });
}

void AppState::start_compression() {
    if (input_items_.empty() || output_archive_path_.empty() || is_working_.load()) return;

    is_working_ = true;
    is_extract_op_ = false;
    operation_finished_ = false;
    operation_failed_ = false;
    cancel_requested_ = false;
    completed_count_ = 0;
    total_count_ = 0;
    total_uncompressed_bytes_ = 0;
    total_compressed_bytes_ = 0;
    elapsed_seconds_ = 0.0;
    mb_per_sec_ = 0.0;
    set_status("Collecting files...");
    start_time_ = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(rows_mutex_);
        rows_.clear();
    }

    if (worker_thread_.joinable()) worker_thread_.join();

    std::vector<std::string> paths;
    paths.reserve(input_items_.size());
    for (const auto& item : input_items_) {
        paths.push_back(item.path);
    }

    const int codec_idx = selected_codec_idx_;
    const int level = compression_level_;
    const std::string out_zip = output_archive_path_;

    worker_thread_ = std::thread([this, paths, codec_idx, level, out_zip]() {
        try {
            auto entries = collect_entries(paths);
            if (entries.empty()) {
                operation_failed_ = true;
                set_status("No files found to compress.");
                is_working_ = false;
                return;
            }

            total_count_ = entries.size();
            {
                std::lock_guard<std::mutex> lock(rows_mutex_);
                rows_.resize(entries.size());
                for (std::size_t i = 0; i < entries.size(); ++i) {
                    rows_[i].name = entries[i].name;
                    rows_[i].uncompressed_size = entries[i].data.size();
                }
            }

            set_status("Compressing " + std::to_string(entries.size()) + " files...");

            auto on_prog = [this](std::size_t done, std::size_t total,
                                  const EntryReport& rep) -> bool {
                completed_count_ = done;
                total_count_ = total;
                total_uncompressed_bytes_ += rep.uncompressed_size;
                total_compressed_bytes_ += rep.compressed_size;

                {
                    std::lock_guard<std::mutex> lock(rows_mutex_);
                    if (rep.index < rows_.size()) {
                        rows_[rep.index].compressed_size = rep.compressed_size;
                        rows_[rep.index].codec = rep.codec;
                        rows_[rep.index].completed = true;
                    }
                }

                return !cancel_requested_.load();
            };

            bool ok = false;
            if (codec_idx == 0) {
                ok = write_zip_auto(out_zip, entries, level, on_prog);
            } else if (codec_idx == 1) {
                ok = write_zip(out_zip, entries, CodecId::Fzac, level, on_prog);
            } else if (codec_idx == 2) {
                ok = write_zip(out_zip, entries, CodecId::Zstd, level, on_prog);
            } else if (codec_idx == 3) {
                ok = write_zip(out_zip, entries, CodecId::Deflate, level, on_prog);
            } else {
                ok = write_zip(out_zip, entries, CodecId::Store, 0, on_prog);
            }

            finish_time_ = std::chrono::steady_clock::now();
            const double dt = std::max(
                0.001, std::chrono::duration<double>(finish_time_ - start_time_).count());
            elapsed_seconds_ = dt;
            mb_per_sec_ =
                (static_cast<double>(total_uncompressed_bytes_.load()) / (1024.0 * 1024.0)) / dt;

            if (cancel_requested_.load()) {
                set_status("Cancelled by user.");
                std::error_code ec;
                fs::remove(out_zip, ec);
            } else if (ok) {
                operation_finished_ = true;
                inspected_archive_path_ = out_zip;
                set_status("Compression complete!");
            } else {
                operation_failed_ = true;
                set_status("Compression failed!");
            }
        } catch (const std::exception& e) {
            operation_failed_ = true;
            finish_time_ = std::chrono::steady_clock::now();
            elapsed_seconds_ =
                std::chrono::duration<double>(finish_time_ - start_time_).count();
            set_status(std::string("Compression error: ") + e.what());
        }
        is_working_ = false;
    });
}

void AppState::cancel_operation() {
    if (is_working_.load()) {
        cancel_requested_ = true;
    }
}

void AppState::render() {
    drain_ipc_paths();

    if (pending_quick_start_) {
        std::chrono::steady_clock::time_point last_t;
        {
            std::lock_guard<std::mutex> lock(ipc_mutex_);
            last_t = last_ipc_time_;
        }
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - last_t).count();
        if (elapsed_ms >= 250) {
            pending_quick_start_ = false;
            stop_ipc_server();
            drain_ipc_paths();
            start_compression();
        }
    }

    if (is_working_.load()) {
        const auto now = std::chrono::steady_clock::now();
        elapsed_seconds_ = std::chrono::duration<double>(now - start_time_).count();
        if (elapsed_seconds_ > 0.02) {
            mb_per_sec_ =
                (static_cast<double>(total_uncompressed_bytes_.load()) / (1024.0 * 1024.0)) /
                elapsed_seconds_;
        }
    }

    if (mode_ == AppMode::QuickProgress) {
        render_quick_progress_view();
        return;
    }

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    constexpr ImGuiWindowFlags kRootFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("FzipMainRoot", nullptr, kRootFlags)) {
        render_header_bar();
        ImGui::Spacing();
        if (mode_ == AppMode::Inspect) {
            render_inspect_view();
        } else {
            render_compress_view();
        }
    }
    ImGui::End();
}

void AppState::render_header_bar() {
    const float h = 50.0f * ui_scale_;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.067f, 0.086f, 0.118f, 1.00f));
    if (ImGui::BeginChild("HeaderBar", ImVec2(0, h), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_NoScrollbar)) {
        draw_brand_badge(26.0f * ui_scale_, mode_ == AppMode::Inspect);
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        if (bold_font_) ImGui::PushFont(bold_font_);
        ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.00f, 1.00f), "FZIP");
        if (bold_font_) ImGui::PopFont();

        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.55f, 0.61f, 0.70f, 1.00f),
                           "  Binary Arithmetic (FZAC) & Zstd Archiver");

        const float tab_w = 105.0f * ui_scale_;
        const float inspect_w = 135.0f * ui_scale_;
        const float ctx_w = 170.0f * ui_scale_;
        const float gap = ImGui::GetStyle().ItemSpacing.x;
        const float right_cluster = tab_w + inspect_w + ctx_w + gap * 2.0f + 12.0f * ui_scale_;
        const float avail = ImGui::GetWindowContentRegionMax().x;
        if (avail > right_cluster + 220.0f * ui_scale_) {
            ImGui::SameLine(avail - right_cluster);
        } else {
            ImGui::SameLine();
        }

        const bool in_compress = (mode_ == AppMode::Compress);
        if (styled_button(
                "Compress", ImVec2(tab_w, 0),
                in_compress ? ImVec4(0.184f, 0.506f, 0.969f, 1.00f)
                            : ImVec4(0.125f, 0.161f, 0.216f, 1.00f),
                ImVec4(0.240f, 0.560f, 0.980f, 1.00f),
                ImVec4(0.145f, 0.392f, 0.780f, 1.00f))) {
            if (!is_working_.load()) mode_ = AppMode::Compress;
        }
        ImGui::SameLine();
        const bool in_inspect = (mode_ == AppMode::Inspect);
        if (styled_button(
                "Inspect Archive", ImVec2(inspect_w, 0),
                in_inspect ? ImVec4(0.184f, 0.506f, 0.969f, 1.00f)
                           : ImVec4(0.125f, 0.161f, 0.216f, 1.00f),
                ImVec4(0.240f, 0.560f, 0.980f, 1.00f),
                ImVec4(0.145f, 0.392f, 0.780f, 1.00f))) {
            if (!is_working_.load()) mode_ = AppMode::Inspect;
        }
        ImGui::SameLine();
        render_context_menu_controls();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void AppState::render_context_menu_controls() {
    const std::string exe = get_exe_path();
    const float btn_w = 165.0f * ui_scale_;

    if (ctx_status_ == ContextMenuStatus::Registered) {
        if (styled_button("Explorer Menu: On", ImVec2(btn_w, 0),
                          ImVec4(0.12f, 0.34f, 0.22f, 1.00f),
                          ImVec4(0.45f, 0.18f, 0.18f, 1.00f),
                          ImVec4(0.55f, 0.15f, 0.15f, 1.00f))) {
            if (unregister_context_menu()) {
                ctx_status_ = ContextMenuStatus::NotRegistered;
                ctx_feedback_ = "Explorer context menu removed.";
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Explorer right-click integration is active.\nClick to unregister.");
        }
    } else {
        if (styled_button("Enable Explorer Menu", ImVec2(btn_w, 0),
                          ImVec4(0.14f, 0.18f, 0.25f, 1.00f),
                          ImVec4(0.18f, 0.42f, 0.28f, 1.00f),
                          ImVec4(0.14f, 0.34f, 0.22f, 1.00f))) {
            if (register_context_menu(exe)) {
                ctx_status_ = ContextMenuStatus::Registered;
                ctx_feedback_ = "Explorer context menu registered!";
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Register right-click Compress / Extract options in Windows Explorer.");
        }
    }
}

void AppState::render_compress_view() {
    const bool working = is_working_.load();

    // Action toolbar
    ImGui::BeginDisabled(working);
    if (styled_button("+ Add Files...", ImVec2(0, 0),
                      ImVec4(0.18f, 0.31f, 0.48f, 1.00f),
                      ImVec4(0.24f, 0.40f, 0.62f, 1.00f),
                      ImVec4(0.15f, 0.26f, 0.42f, 1.00f))) {
        add_input_paths(pick_files("Add Files to Compress"));
    }
    ImGui::SameLine();
    if (ImGui::Button("+ Add Folder...")) {
        auto folder = pick_folder("Add Folder to Compress");
        if (!folder.empty()) add_input_paths({folder});
    }
    ImGui::SameLine();
    if (ImGui::Button("Open ZIP...")) {
        auto z = pick_file("Open ZIP Archive");
        if (!z.empty()) open_archive(z);
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear List")) {
        input_items_.clear();
        output_archive_path_.clear();
        auto_output_path_ = true;
        std::lock_guard<std::mutex> lock(rows_mutex_);
        rows_.clear();
    }
    ImGui::EndDisabled();

    std::uint64_t queued_bytes = 0;
    for (const auto& it : input_items_) queued_bytes += it.size;
    {
        char summary[96];
        std::snprintf(summary, sizeof(summary), "%zu items  |  %s",
                      input_items_.size(), format_size(queued_bytes).c_str());
        const float tw = ImGui::CalcTextSize(summary).x;
        const float rx = ImGui::GetWindowContentRegionMax().x - tw - 8.0f * ui_scale_;
        if (rx > ImGui::GetCursorPosX() + 180.0f) {
            ImGui::SameLine(rx);
        } else {
            ImGui::SameLine();
        }
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(ImVec4(0.62f, 0.68f, 0.78f, 1.00f), "%s", summary);
    }

    ImGui::Spacing();

    // Compression settings card
    const float settings_h = 84.0f * ui_scale_;
    if (ImGui::BeginChild("SettingsCard", ImVec2(0, settings_h), ImGuiChildFlags_Borders)) {
        ImGui::BeginDisabled(working);

        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(ImVec4(0.65f, 0.72f, 0.82f, 1.00f), "Archive:");
        ImGui::SameLine();
        char buf[1024] = {0};
        std::snprintf(buf, sizeof(buf), "%s", output_archive_path_.c_str());
        const float browse_w = 96.0f * ui_scale_;
        ImGui::SetNextItemWidth(std::max(120.0f, ImGui::GetContentRegionAvail().x - browse_w - 10.0f));
        if (ImGui::InputText("##outpath", buf, sizeof(buf))) {
            output_archive_path_ = buf;
            auto_output_path_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse...", ImVec2(browse_w, 0))) {
            auto s = pick_save_file(
                output_archive_path_.empty() ? "archive.zip" : output_archive_path_);
            if (!s.empty()) {
                output_archive_path_ = s;
                auto_output_path_ = false;
            }
        }

        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(ImVec4(0.65f, 0.72f, 0.82f, 1.00f), "Codec:  ");
        ImGui::SameLine();
        const char* codecs[] = {
            "Auto (Best of Zstd / Deflate / Store)",
            "FZAC (Binary Arithmetic + BWT/LZP)",
            "Zstd",
            "Deflate",
            "Store (No Compression)"
        };
        ImGui::SetNextItemWidth(300.0f * ui_scale_);
        ImGui::Combo("##codec", &selected_codec_idx_, codecs, IM_ARRAYSIZE(codecs));
        ImGui::SameLine();

        const bool is_22 = (selected_codec_idx_ == 0 || selected_codec_idx_ == 2);
        const bool is_12 = (selected_codec_idx_ == 1 || selected_codec_idx_ == 3);
        if (is_22 || is_12) {
            const int max_lvl = is_22 ? 22 : 12;
            if (compression_level_ > max_lvl) compression_level_ = is_22 ? 19 : 9;
            ImGui::SetNextItemWidth(175.0f * ui_scale_);
            ImGui::SliderInt("##level", &compression_level_, 1, max_lvl,
                             is_22 ? "Level %d / 22" : "Level %d / 12");
            ImGui::SameLine();
            if (ImGui::Button("Fast")) compression_level_ = 3;
            ImGui::SameLine();
            if (ImGui::Button("Balanced")) compression_level_ = is_22 ? 9 : 6;
            ImGui::SameLine();
            if (ImGui::Button("Ultra")) compression_level_ = max_lvl;
        } else {
            ImGui::TextDisabled("Uncompressed pass-through (maximum speed)");
        }

        ImGui::EndDisabled();
    }
    ImGui::EndChild();

    ImGui::Spacing();

    // Middle files / dropzone card
    const float footer_h = 116.0f * ui_scale_;
    const float body_h = std::max(140.0f * ui_scale_, ImGui::GetContentRegionAvail().y - footer_h - 8.0f);

    if (ImGui::BeginChild("FileListCard", ImVec2(0, body_h), ImGuiChildFlags_Borders)) {
        std::lock_guard<std::mutex> lock(rows_mutex_);
        if (input_items_.empty() && rows_.empty()) {
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            ImGui::SetCursorPosY(std::max(16.0f, avail.y * 0.34f));
            const char* hint1 = "Drag & Drop Files or Folders Here";
            const char* hint2 = "Auto mode picks the smallest output across standard Zstd, Deflate, and Store";
            const float w1 = ImGui::CalcTextSize(hint1).x;
            const float w2 = ImGui::CalcTextSize(hint2).x;
            ImGui::SetCursorPosX(std::max(10.0f, (avail.x - w1) * 0.5f));
            if (bold_font_) ImGui::PushFont(bold_font_);
            ImGui::TextColored(ImVec4(0.75f, 0.82f, 0.92f, 1.00f), "%s", hint1);
            if (bold_font_) ImGui::PopFont();
            ImGui::SetCursorPosX(std::max(10.0f, (avail.x - w2) * 0.5f));
            ImGui::TextDisabled("%s", hint2);
        } else {
            const ImGuiTableFlags tflags =
                ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
            if (ImGui::BeginTable("FilesTable", 6, tflags, ImGui::GetContentRegionAvail())) {
                ImGui::TableSetupColumn("Name / Path", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Original", ImGuiTableColumnFlags_WidthFixed, 92.0f * ui_scale_);
                ImGui::TableSetupColumn("Compressed", ImGuiTableColumnFlags_WidthFixed, 96.0f * ui_scale_);
                ImGui::TableSetupColumn("Ratio", ImGuiTableColumnFlags_WidthFixed, 72.0f * ui_scale_);
                ImGui::TableSetupColumn("Codec", ImGuiTableColumnFlags_WidthFixed, 76.0f * ui_scale_);
                ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 68.0f * ui_scale_);
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableHeadersRow();

                if (rows_.empty()) {
                    int remove_idx = -1;
                    for (int idx = 0; idx < static_cast<int>(input_items_.size()); ++idx) {
                        const auto& item = input_items_[static_cast<std::size_t>(idx)];
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("%s%s", item.is_directory ? "[DIR] " : "", item.path.c_str());
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%s", format_size(item.size).c_str());
                        ImGui::TableSetColumnIndex(2);
                        ImGui::TextDisabled("-");
                        ImGui::TableSetColumnIndex(3);
                        ImGui::TextDisabled("-");
                        ImGui::TableSetColumnIndex(4);
                        ImGui::TextDisabled("Queued");
                        ImGui::TableSetColumnIndex(5);
                        if (!working) {
                            ImGui::PushID(idx);
                            if (ImGui::SmallButton("Remove")) remove_idx = idx;
                            ImGui::PopID();
                        }
                    }
                    if (remove_idx >= 0) {
                        input_items_.erase(input_items_.begin() + remove_idx);
                    }
                } else {
                    for (const auto& r : rows_) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("%s", r.name.c_str());
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%s", format_size(r.uncompressed_size).c_str());
                        ImGui::TableSetColumnIndex(2);
                        if (r.completed) {
                            ImGui::Text("%s", format_size(r.compressed_size).c_str());
                        } else {
                            ImGui::TextDisabled("...");
                        }
                        ImGui::TableSetColumnIndex(3);
                        if (r.completed && r.uncompressed_size > 0) {
                            const double ratio = static_cast<double>(r.uncompressed_size) /
                                                 (r.compressed_size ? r.compressed_size : 1);
                            ImGui::TextColored(
                                ratio >= 2.0 ? ImVec4(0.35f, 0.88f, 0.55f, 1.0f)
                                             : ImVec4(0.85f, 0.90f, 0.96f, 1.0f),
                                "%.2fx", ratio);
                        } else {
                            ImGui::TextDisabled("-");
                        }
                        ImGui::TableSetColumnIndex(4);
                        if (r.completed) {
                            ImGui::TextColored(codec_color(r.codec), "%s", codec_name(r.codec));
                        } else {
                            ImGui::TextDisabled("-");
                        }
                        ImGui::TableSetColumnIndex(5);
                        if (r.completed) {
                            ImGui::TextColored(ImVec4(0.35f, 0.88f, 0.55f, 1.0f), "Done");
                        } else {
                            ImGui::TextColored(ImVec4(0.95f, 0.78f, 0.30f, 1.0f), "Active");
                        }
                    }
                }
                ImGui::EndTable();
            }
        }
    }
    ImGui::EndChild();

    ImGui::Spacing();

    // Footer telemetry & primary action card
    if (ImGui::BeginChild("FooterCard", ImVec2(0, 0), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_NoScrollbar)) {
        const std::size_t done = completed_count_.load();
        const std::size_t total = total_count_.load();
        const float fraction = (total > 0) ? (static_cast<float>(done) / static_cast<float>(total)) : 0.0f;
        const std::uint64_t raw_done = total_uncompressed_bytes_.load();
        const std::uint64_t comp_done = total_compressed_bytes_.load();

        if (ImGui::BeginTable("MetricsGrid", 4)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            char files_str[64];
            std::snprintf(files_str, sizeof(files_str), "%zu / %zu", done, total);
            render_metric_card("Files Processed", files_str, ImVec4(0.92f, 0.94f, 0.97f, 1.0f));

            ImGui::TableSetColumnIndex(1);
            char spd_str[64];
            std::snprintf(spd_str, sizeof(spd_str), "%.1f MB/s", mb_per_sec_);
            render_metric_card("Throughput", spd_str, ImVec4(0.35f, 0.65f, 1.00f, 1.0f));

            ImGui::TableSetColumnIndex(2);
            char time_str[64];
            std::snprintf(time_str, sizeof(time_str), "%.2f s", elapsed_seconds_);
            render_metric_card("Elapsed", time_str, ImVec4(0.92f, 0.94f, 0.97f, 1.0f));

            ImGui::TableSetColumnIndex(3);
            if (raw_done > 0 && comp_done > 0) {
                const double ratio = static_cast<double>(raw_done) / static_cast<double>(comp_done);
                char ratio_str[96];
                std::snprintf(ratio_str, sizeof(ratio_str), "%s -> %s (%.2fx)",
                              format_size(raw_done).c_str(),
                              format_size(comp_done).c_str(), ratio);
                render_metric_card("Compression Result", ratio_str,
                                   ImVec4(0.28f, 0.88f, 0.52f, 1.00f));
            } else {
                render_metric_card("Compression Result", "Ready",
                                   ImVec4(0.56f, 0.62f, 0.71f, 1.00f));
            }
            ImGui::EndTable();
        }

        const float btn_w = 150.0f * ui_scale_;
        const float inspect_btn_w = 128.0f * ui_scale_;
        const bool show_inspect = !working && operation_finished_.load() && !is_extract_op_.load() &&
                                  !inspected_archive_path_.empty();
        const float bar_w = std::max(
            120.0f, ImGui::GetContentRegionAvail().x - btn_w -
                        (show_inspect ? (inspect_btn_w + 16.0f) : 10.0f));

        char overlay[128];
        if (total > 0) {
            std::snprintf(overlay, sizeof(overlay), "%.0f%%  (%zu / %zu)",
                          fraction * 100.0f, done, total);
        } else {
            std::snprintf(overlay, sizeof(overlay), "%s",
                          get_status().empty() ? "Idle" : get_status().c_str());
        }
        ImGui::ProgressBar(fraction, ImVec2(bar_w, 30.0f * ui_scale_), overlay);
        ImGui::SameLine();

        if (working) {
            if (styled_button("Cancel", ImVec2(btn_w, 30.0f * ui_scale_),
                              ImVec4(0.75f, 0.20f, 0.22f, 1.00f),
                              ImVec4(0.86f, 0.25f, 0.27f, 1.00f),
                              ImVec4(0.64f, 0.16f, 0.18f, 1.00f))) {
                cancel_operation();
            }
        } else {
            ImGui::BeginDisabled(input_items_.empty() || output_archive_path_.empty());
            if (styled_button("Compress Now", ImVec2(btn_w, 30.0f * ui_scale_),
                              ImVec4(0.137f, 0.525f, 0.212f, 1.00f),
                              ImVec4(0.180f, 0.627f, 0.263f, 1.00f),
                              ImVec4(0.110f, 0.431f, 0.173f, 1.00f))) {
                start_compression();
            }
            ImGui::EndDisabled();
            if (show_inspect) {
                ImGui::SameLine();
                if (ImGui::Button("Inspect Result", ImVec2(inspect_btn_w, 30.0f * ui_scale_))) {
                    open_archive(inspected_archive_path_);
                }
            }
        }

        if (!ctx_feedback_.empty()) {
            ImGui::TextColored(ImVec4(0.42f, 0.78f, 0.98f, 1.00f), "%s", ctx_feedback_.c_str());
        } else {
            const std::string st = get_status();
            if (!st.empty() && operation_failed_.load()) {
                ImGui::TextColored(ImVec4(0.95f, 0.38f, 0.38f, 1.00f), "%s", st.c_str());
            }
        }
    }
    ImGui::EndChild();
}

void AppState::render_inspect_view() {
    const bool working = is_working_.load();

    if (inspected_archive_path_.empty()) {
        if (ImGui::BeginChild("EmptyInspect", ImVec2(0, 0), ImGuiChildFlags_Borders)) {
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            ImGui::SetCursorPosY(std::max(40.0f, avail.y * 0.36f));
            const char* msg = "No archive opened";
            const float mw = ImGui::CalcTextSize(msg).x;
            ImGui::SetCursorPosX(std::max(10.0f, (avail.x - mw) * 0.5f));
            if (bold_font_) ImGui::PushFont(bold_font_);
            ImGui::TextColored(ImVec4(0.75f, 0.82f, 0.92f, 1.00f), "%s", msg);
            if (bold_font_) ImGui::PopFont();

            const float bw = 180.0f * ui_scale_;
            ImGui::Spacing();
            ImGui::SetCursorPosX(std::max(10.0f, (avail.x - bw) * 0.5f));
            if (styled_button("Open ZIP Archive...", ImVec2(bw, 34.0f * ui_scale_),
                              ImVec4(0.184f, 0.506f, 0.969f, 1.00f),
                              ImVec4(0.240f, 0.560f, 0.980f, 1.00f),
                              ImVec4(0.145f, 0.392f, 0.780f, 1.00f))) {
                auto z = pick_file("Open ZIP Archive");
                if (!z.empty()) open_archive(z);
            }
        }
        ImGui::EndChild();
        return;
    }

    // Archive summary & actions card
    const float top_h = 92.0f * ui_scale_;
    if (ImGui::BeginChild("InspectHeaderCard", ImVec2(0, top_h), ImGuiChildFlags_Borders)) {
        const fs::path arch_p(inspected_archive_path_);
        const double ratio = (inspected_total_comp_ > 0)
            ? (static_cast<double>(inspected_total_uncomp_) / static_cast<double>(inspected_total_comp_))
            : 1.0;

        if (bold_font_) ImGui::PushFont(bold_font_);
        ImGui::Text("%s", arch_p.filename().string().c_str());
        if (bold_font_) ImGui::PopFont();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", inspected_archive_path_.c_str());
        }

        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.58f, 0.65f, 0.75f, 1.00f),
                           "  (%zu entries | %s -> %s | %.2fx)",
                           inspected_entries_.size(),
                           format_size(inspected_total_uncomp_).c_str(),
                           format_size(inspected_total_comp_).c_str(),
                           ratio);

        ImGui::BeginDisabled(working);
        ImGui::SetNextItemWidth(240.0f * ui_scale_);
        ImGui::InputTextWithHint("##filter", "Filter entries...", inspect_filter_.data(),
                                 inspect_filter_.size());
        ImGui::SameLine();

        if (ImGui::Button("Open Another...")) {
            auto z = pick_file("Open ZIP Archive");
            if (!z.empty()) open_archive(z);
        }
        ImGui::SameLine();

        if (styled_button("Extract to Subfolder", ImVec2(0, 0),
                          ImVec4(0.137f, 0.525f, 0.212f, 1.00f),
                          ImVec4(0.180f, 0.627f, 0.263f, 1.00f),
                          ImVec4(0.110f, 0.431f, 0.173f, 1.00f))) {
            fs::path sub_dir = arch_p.parent_path() / arch_p.stem();
            extract_archive(sub_dir.string());
        }
        ImGui::SameLine();
        if (styled_button("Extract All To...", ImVec2(0, 0),
                          ImVec4(0.184f, 0.506f, 0.969f, 1.00f),
                          ImVec4(0.240f, 0.560f, 0.980f, 1.00f),
                          ImVec4(0.145f, 0.392f, 0.780f, 1.00f))) {
            auto out_dir = pick_folder("Select Destination Folder");
            if (!out_dir.empty()) extract_archive(out_dir);
        }
        if (!working && !last_extracted_dir_.empty()) {
            ImGui::SameLine();
            if (ImGui::Button("Open Folder")) {
                open_folder_in_explorer(last_extracted_dir_);
            }
        }
        ImGui::EndDisabled();
    }
    ImGui::EndChild();

    ImGui::Spacing();

    const float status_bar_h = 60.0f * ui_scale_;
    const float table_h = std::max(140.0f * ui_scale_, ImGui::GetContentRegionAvail().y - status_bar_h - 8.0f);

    if (ImGui::BeginChild("InspectTableCard", ImVec2(0, table_h), ImGuiChildFlags_Borders)) {
        const ImGuiTableFlags tflags =
            ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
        if (ImGui::BeginTable("InspectTable", 6, tflags, ImGui::GetContentRegionAvail())) {
            ImGui::TableSetupColumn("Entry Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Original", ImGuiTableColumnFlags_WidthFixed, 96.0f * ui_scale_);
            ImGui::TableSetupColumn("Compressed", ImGuiTableColumnFlags_WidthFixed, 96.0f * ui_scale_);
            ImGui::TableSetupColumn("Ratio", ImGuiTableColumnFlags_WidthFixed, 72.0f * ui_scale_);
            ImGui::TableSetupColumn("Codec", ImGuiTableColumnFlags_WidthFixed, 76.0f * ui_scale_);
            ImGui::TableSetupColumn("CRC-32", ImGuiTableColumnFlags_WidthFixed, 88.0f * ui_scale_);
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();

            const std::string_view filter(inspect_filter_.data());
            for (const auto& e : inspected_entries_) {
                if (!icontains(e.name, filter)) continue;
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%s", e.name.c_str());
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%s", format_size(e.uncompressed_size).c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%s", format_size(e.compressed_size).c_str());
                ImGui::TableSetColumnIndex(3);
                const double r = static_cast<double>(e.uncompressed_size) /
                                 (e.compressed_size ? e.compressed_size : 1);
                ImGui::Text("%.2fx", r);
                ImGui::TableSetColumnIndex(4);
                ImGui::TextColored(codec_color(e.codec), "%s", codec_name(e.codec));
                ImGui::TableSetColumnIndex(5);
                ImGui::TextDisabled("%08X", e.crc32);
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();

    ImGui::Spacing();

    if (ImGui::BeginChild("InspectFooterCard", ImVec2(0, 0), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_NoScrollbar)) {
        const std::size_t tot = total_count_.load();
        const std::size_t done = completed_count_.load();
        const float frac = (tot > 0) ? (static_cast<float>(done) / static_cast<float>(tot)) : 0.0f;

        if (working) {
            ImGui::ProgressBar(frac, ImVec2(260.0f * ui_scale_, 24.0f * ui_scale_));
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.78f, 0.30f, 1.00f), "%s", get_status().c_str());
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) cancel_operation();
        } else {
            const std::string st = get_status();
            const ImVec4 col = operation_failed_.load()
                ? ImVec4(0.95f, 0.38f, 0.38f, 1.00f)
                : (operation_finished_.load() ? ImVec4(0.28f, 0.88f, 0.52f, 1.00f)
                                              : ImVec4(0.65f, 0.72f, 0.82f, 1.00f));
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(col, "%s", st.c_str());
        }
    }
    ImGui::EndChild();
}

void AppState::render_quick_progress_view() {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

    constexpr ImGuiWindowFlags kQuickFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("QuickProgressView", nullptr, kQuickFlags)) {
        if (ImGui::BeginChild("QuickCard", ImVec2(0, 0), ImGuiChildFlags_Borders)) {
            const bool working = is_working_.load();
            const bool extracting = is_extract_op_.load();
            const bool failed = operation_failed_.load();

            draw_brand_badge(24.0f * ui_scale_, extracting);
            ImGui::SameLine();
            if (bold_font_) ImGui::PushFont(bold_font_);
            ImGui::TextColored(extracting ? ImVec4(0.28f, 0.88f, 0.62f, 1.00f)
                                          : ImVec4(0.35f, 0.65f, 1.00f, 1.00f),
                               extracting ? "FZIP  |  Extracting Archive"
                                          : "FZIP  |  Compressing Archive");
            if (bold_font_) ImGui::PopFont();

            const std::string target_label = extracting
                ? (last_extracted_dir_.empty() ? inspected_archive_path_ : last_extracted_dir_)
                : output_archive_path_;
            if (!target_label.empty()) {
                ImGui::TextDisabled("%s", target_label.c_str());
            }

            ImGui::Separator();
            ImGui::Spacing();

            const std::string st = get_status();
            const ImVec4 st_col = failed
                ? ImVec4(0.95f, 0.38f, 0.38f, 1.00f)
                : (working ? ImVec4(0.95f, 0.78f, 0.30f, 1.00f)
                           : ImVec4(0.28f, 0.88f, 0.52f, 1.00f));
            ImGui::TextWrapped("%s", st.c_str());
            if (!failed && !working) {
                ImGui::SameLine();
                ImGui::TextColored(st_col, " (OK)");
            }
            ImGui::Spacing();

            const std::size_t done = completed_count_.load();
            const std::size_t total = total_count_.load();
            const float fraction =
                (total > 0) ? (static_cast<float>(done) / static_cast<float>(total)) : 0.0f;

            char bar_buf[128];
            if (total > 0) {
                std::snprintf(bar_buf, sizeof(bar_buf), "%.0f%%  (%zu / %zu  |  %.1f MB/s)",
                              fraction * 100.0f, done, total, mb_per_sec_);
            } else {
                std::snprintf(bar_buf, sizeof(bar_buf), working ? "Working..." : "Done");
            }
            ImGui::ProgressBar(working && total == 0 ? 0.25f : fraction,
                               ImVec2(-1, 28.0f * ui_scale_), bar_buf);

            ImGui::Spacing();
            ImGui::TextDisabled("Processed: %s   |   Elapsed: %.2f s   |   Speed: %.1f MB/s",
                                format_size(total_uncompressed_bytes_.load()).c_str(),
                                elapsed_seconds_, mb_per_sec_);

            ImGui::Spacing();
            if (working) {
                if (styled_button("Cancel", ImVec2(110.0f * ui_scale_, 30.0f * ui_scale_),
                                  ImVec4(0.75f, 0.20f, 0.22f, 1.00f),
                                  ImVec4(0.86f, 0.25f, 0.27f, 1.00f),
                                  ImVec4(0.64f, 0.16f, 0.18f, 1.00f))) {
                    cancel_operation();
                    close_requested_ = true;
                }
            } else {
                if (operation_finished_.load() && !failed && !cancel_requested_.load()) {
                    const auto now = std::chrono::steady_clock::now();
                    const double finish_elapsed =
                        std::chrono::duration<double>(now - finish_time_).count();
                    if (finish_elapsed > 1.4) {
                        close_requested_ = true;
                    }
                }
                if (extracting && !failed && !last_extracted_dir_.empty()) {
                    if (ImGui::Button("Open Folder", ImVec2(120.0f * ui_scale_, 30.0f * ui_scale_))) {
                        open_folder_in_explorer(last_extracted_dir_);
                    }
                    ImGui::SameLine();
                }
                if (ImGui::Button("Close", ImVec2(110.0f * ui_scale_, 30.0f * ui_scale_))) {
                    close_requested_ = true;
                }
            }
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

}  // namespace fzip::gui
