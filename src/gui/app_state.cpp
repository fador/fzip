#include "app_state.hpp"
#include "native_dialogs.hpp"
#include "zip_reader.hpp"
#include "zip_writer.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <imgui.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fzip::gui {

namespace fs = std::filesystem;

namespace {

std::string format_size(std::uint64_t bytes) {
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

const char* codec_name(CodecId id) {
    switch (id) {
        case CodecId::Store: return "Store";
        case CodecId::Deflate: return "Deflate";
        case CodecId::Zstd: return "Zstd";
    }
    return "Unknown";
}

}  // namespace

AppState::AppState() {
    ctx_status_ = check_context_menu_status();
}

AppState::~AppState() {
    cancel_operation();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void AppState::init(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    if (args.empty()) {
        mode_ = AppMode::Compress;
        return;
    }

    std::string_view first = args[0];

    if (first == "--register-context-menu") {
        char exe_buf[1024] = {0};
#if defined(_WIN32)
        GetModuleFileNameA(nullptr, exe_buf, sizeof(exe_buf));
#else
        std::string p = fs::canonical("/proc/self/exe").string();
        std::snprintf(exe_buf, sizeof(exe_buf), "%s", p.c_str());
#endif
        register_context_menu(exe_buf);
        close_requested_ = true;
        return;
    }

    if (first == "--unregister-context-menu") {
        unregister_context_menu();
        close_requested_ = true;
        return;
    }

    if (first == "--quick-compress") {
        mode_ = AppMode::QuickProgress;
        std::vector<std::string> input_paths(args.begin() + 1, args.end());
        add_input_paths(input_paths);
        start_compression();
        return;
    }

    if (first == "--compress-dialog") {
        mode_ = AppMode::Compress;
        std::vector<std::string> input_paths(args.begin() + 1, args.end());
        add_input_paths(input_paths);
        return;
    }

    if (first == "--extract-here" && args.size() >= 2) {
        mode_ = AppMode::QuickProgress;
        fs::path zip_path(args[1]);
        extract_archive(zip_path.parent_path().string());
        return;
    }

    if (first == "--extract-subdir" && args.size() >= 2) {
        mode_ = AppMode::QuickProgress;
        fs::path zip_path(args[1]);
        fs::path out_dir = zip_path.parent_path() / zip_path.stem();
        extract_archive(out_dir.string());
        return;
    }

    // Direct path passed: check if single .zip
    if (args.size() == 1) {
        fs::path p(args[0]);
        if (p.extension() == ".zip") {
            open_archive(args[0]);
            return;
        }
    }

    // Default: treated as inputs to compress
    mode_ = AppMode::Compress;
    add_input_paths(args);
}

void AppState::add_input_paths(const std::vector<std::string>& paths) {
    if (paths.empty()) return;
    for (const auto& p_str : paths) {
        fs::path p(p_str);
        std::error_code ec;
        if (!fs::exists(p, ec)) continue;

        FileItem item;
        item.path = fs::absolute(p).string();
        item.is_directory = fs::is_directory(p, ec);
        if (!item.is_directory) {
            item.size = fs::file_size(p, ec);
        } else {
            // Count approximate directory size
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

    // Auto suggest output archive path if currently empty
    if (output_archive_path_.empty() && !input_items_.empty()) {
        fs::path first(input_items_[0].path);
        fs::path dest = first.parent_path() / (first.stem().string() + ".zip");
        output_archive_path_ = dest.string();
    }
}

void AppState::open_archive(const std::string& path) {
    inspected_archive_path_ = path;
    inspected_entries_ = read_zip_entries(path);
    inspected_total_uncomp_ = 0;
    inspected_total_comp_ = 0;
    for (const auto& e : inspected_entries_) {
        inspected_total_uncomp_ += e.uncompressed_size;
        inspected_total_comp_ += e.compressed_size;
    }
    mode_ = AppMode::Inspect;
}

void AppState::extract_archive(const std::string& out_dir) {
    is_working_ = true;
    cancel_requested_ = false;
    status_message_ = "Extracting to " + out_dir + "...";
    start_time_ = std::chrono::steady_clock::now();

    if (worker_thread_.joinable()) worker_thread_.join();

    worker_thread_ = std::thread([this, out_dir]() {
        try {
            fs::create_directories(out_dir);
            auto entries = extract_all(inspected_archive_path_.empty() ? output_archive_path_ : inspected_archive_path_);
            total_count_ = entries.size();
            for (std::size_t i = 0; i < entries.size(); ++i) {
                if (cancel_requested_.load()) break;
                const auto& [name, data] = entries[i];
                fs::path p = fs::path(out_dir) / name;
                if (p.has_parent_path()) {
                    fs::create_directories(p.parent_path());
                }
                std::ofstream out(p, std::ios::binary);
                if (out && !data.empty()) {
                    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
                }
                completed_count_ = i + 1;
            }
            status_message_ = "Extraction completed.";
        } catch (const std::exception& e) {
            status_message_ = std::string("Extraction error: ") + e.what();
        }
        is_working_ = false;
    });
}

void AppState::start_compression() {
    if (input_items_.empty() || output_archive_path_.empty() || is_working_) return;

    is_working_ = true;
    cancel_requested_ = false;
    completed_count_ = 0;
    total_count_ = 0;
    total_uncompressed_bytes_ = 0;
    total_compressed_bytes_ = 0;
    status_message_ = "Collecting files...";
    start_time_ = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(rows_mutex_);
        rows_.clear();
    }

    if (worker_thread_.joinable()) worker_thread_.join();

    std::vector<std::string> paths;
    for (const auto& item : input_items_) {
        paths.push_back(item.path);
    }

    int codec_idx = selected_codec_idx_;
    int level = compression_level_;
    std::string out_zip = output_archive_path_;

    worker_thread_ = std::thread([this, paths, codec_idx, level, out_zip]() {
        auto entries = collect_entries(paths);
        if (entries.empty()) {
            status_message_ = "No files found to compress.";
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

        status_message_ = "Compressing...";

        auto on_prog = [this](std::size_t done, std::size_t total, const EntryReport& rep) -> bool {
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
            ok = write_zip(out_zip, entries, CodecId::Zstd, level, on_prog);
        } else if (codec_idx == 2) {
            ok = write_zip(out_zip, entries, CodecId::Deflate, level, on_prog);
        } else {
            ok = write_zip(out_zip, entries, CodecId::Store, 0, on_prog);
        }

        if (cancel_requested_.load()) {
            status_message_ = "Cancelled by user.";
            std::error_code ec;
            fs::remove(out_zip, ec);
        } else if (ok) {
            status_message_ = "Compression complete!";
        } else {
            status_message_ = "Compression failed!";
        }
        finish_time_ = std::chrono::steady_clock::now();
        operation_finished_ = true;
        is_working_ = false;
    });
}

void AppState::cancel_operation() {
    if (is_working_) {
        cancel_requested_ = true;
    }
}

void AppState::render() {
    // Update live metrics
    if (is_working_) {
        auto now = std::chrono::steady_clock::now();
        elapsed_seconds_ = std::chrono::duration<double>(now - start_time_).count();
        if (elapsed_seconds_ > 0.05) {
            mb_per_sec_ = (total_uncompressed_bytes_ / (1024.0 * 1024.0)) / elapsed_seconds_;
        }
    }

    if (mode_ == AppMode::QuickProgress) {
        render_quick_progress_view();
    } else if (mode_ == AppMode::Inspect) {
        render_inspect_view();
    } else {
        render_compress_view();
    }
}

void AppState::render_compress_view() {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("CompressorView", nullptr, flags)) {
        // Top toolbar
        if (ImGui::Button("Add Files...")) {
            auto files = pick_files("Add Files to Compress");
            add_input_paths(files);
        }
        ImGui::SameLine();
        if (ImGui::Button("Add Folder...")) {
            auto folder = pick_folder("Add Folder to Compress");
            if (!folder.empty()) {
                add_input_paths({folder});
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear Items")) {
            if (!is_working_) {
                input_items_.clear();
                std::lock_guard<std::mutex> lock(rows_mutex_);
                rows_.clear();
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Open Existing ZIP...")) {
            auto z = pick_file("Open ZIP Archive");
            if (!z.empty()) {
                open_archive(z);
            }
        }

        ImGui::Separator();

        // Output destination & settings
        ImGui::Text("Save to:");
        ImGui::SameLine();
        char buf[1024] = {0};
        std::snprintf(buf, sizeof(buf), "%s", output_archive_path_.c_str());
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
        if (ImGui::InputText("##outpath", buf, sizeof(buf))) {
            output_archive_path_ = buf;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse...")) {
            auto s = pick_save_file(output_archive_path_.empty() ? "archive.zip" : output_archive_path_);
            if (!s.empty()) {
                output_archive_path_ = s;
            }
        }

        // Codec & Level selection
        const char* codecs[] = { "Auto (SOTA Best of Zstd/Deflate/Store)", "Zstd", "Deflate", "Store (No Compression)" };
        ImGui::SetNextItemWidth(260);
        ImGui::Combo("Codec", &selected_codec_idx_, codecs, IM_ARRAYSIZE(codecs));
        ImGui::SameLine();

        if (selected_codec_idx_ == 0 || selected_codec_idx_ == 1) {
            ImGui::SetNextItemWidth(180);
            ImGui::SliderInt("Level", &compression_level_, 1, 22, "%d (22 = Optimal)");
        } else if (selected_codec_idx_ == 2) {
            if (compression_level_ > 12) compression_level_ = 9;
            ImGui::SetNextItemWidth(180);
            ImGui::SliderInt("Level", &compression_level_, 1, 12, "%d (12 = Optimal)");
        }

        ImGui::Separator();

        // Items table
        float table_height = ImGui::GetContentRegionAvail().y - 120;
        if (table_height < 100) table_height = 100;

        if (ImGui::BeginTable("FilesTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, table_height))) {
            ImGui::TableSetupColumn("Name / Path", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Original Size", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("Compressed", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("Ratio", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Codec", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableHeadersRow();

            std::lock_guard<std::mutex> lock(rows_mutex_);
            if (rows_.empty()) {
                for (const auto& item : input_items_) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%s", item.path.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%s", format_size(item.size).c_str());
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("-");
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("-");
                    ImGui::TableSetColumnIndex(4);
                    ImGui::Text("-");
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
                        ImGui::Text("...");
                    }
                    ImGui::TableSetColumnIndex(3);
                    if (r.completed && r.uncompressed_size > 0) {
                        double ratio = static_cast<double>(r.uncompressed_size) / (r.compressed_size ? r.compressed_size : 1);
                        ImGui::Text("%.2fx", ratio);
                    } else {
                        ImGui::Text("-");
                    }
                    ImGui::TableSetColumnIndex(4);
                    if (r.completed) {
                        ImGui::Text("%s", codec_name(r.codec));
                    } else {
                        ImGui::Text("-");
                    }
                }
            }
            ImGui::EndTable();
        }

        ImGui::Separator();

        // Bottom progress & action bar
        std::size_t done = completed_count_.load();
        std::size_t total = total_count_.load();
        float fraction = (total > 0) ? (static_cast<float>(done) / total) : 0.0f;

        char prog_overlay[128] = {0};
        std::snprintf(prog_overlay, sizeof(prog_overlay), "%zu / %zu files (%.1f MB/s)", done, total, mb_per_sec_);
        ImGui::ProgressBar(fraction, ImVec2(-1, 24), prog_overlay);

        ImGui::Spacing();
        if (is_working_) {
            if (ImGui::Button("Cancel", ImVec2(120, 32))) {
                cancel_operation();
            }
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "%s", status_message_.c_str());
        } else {
            if (ImGui::Button("Compress Now", ImVec2(140, 32))) {
                start_compression();
            }
            ImGui::SameLine();
            if (!status_message_.empty()) {
                ImGui::Text("%s", status_message_.c_str());
            }
            if (total_uncompressed_bytes_ > 0 && total_compressed_bytes_ > 0) {
                ImGui::SameLine();
                double total_ratio = static_cast<double>(total_uncompressed_bytes_) / total_compressed_bytes_;
                ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f),
                                   " | Saved %s (Ratio: %.2fx)",
                                   format_size(total_uncompressed_bytes_ - total_compressed_bytes_).c_str(),
                                   total_ratio);
            }
        }

        render_context_menu_settings();
    }
    ImGui::End();
}

void AppState::render_inspect_view() {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("InspectView", nullptr, flags)) {
        if (ImGui::Button("< Back to Compressor")) {
            mode_ = AppMode::Compress;
        }
        ImGui::SameLine();
        ImGui::Text("Archive: %s (%zu entries)", inspected_archive_path_.c_str(), inspected_entries_.size());
        ImGui::SameLine();
        if (ImGui::Button("Extract All To...")) {
            auto out_dir = pick_folder("Select Destination Folder");
            if (!out_dir.empty()) {
                extract_archive(out_dir);
            }
        }

        ImGui::Separator();

        float table_height = ImGui::GetContentRegionAvail().y - 40;
        if (ImGui::BeginTable("InspectTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, table_height))) {
            ImGui::TableSetupColumn("Entry Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Original Size", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("Compressed Size", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("Ratio", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Codec", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("CRC-32", ImGuiTableColumnFlags_WidthFixed, 90);
            ImGui::TableHeadersRow();

            for (const auto& e : inspected_entries_) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%s", e.name.c_str());
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%s", format_size(e.uncompressed_size).c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%s", format_size(e.compressed_size).c_str());
                ImGui::TableSetColumnIndex(3);
                double ratio = static_cast<double>(e.uncompressed_size) / (e.compressed_size ? e.compressed_size : 1);
                ImGui::Text("%.2fx", ratio);
                ImGui::TableSetColumnIndex(4);
                ImGui::Text("%s", codec_name(e.codec));
                ImGui::TableSetColumnIndex(5);
                ImGui::Text("%08X", e.crc32);
            }
            ImGui::EndTable();
        }

        if (is_working_) {
            std::size_t tot = total_count_.load();
            float frac = (tot > 0) ? (static_cast<float>(completed_count_.load()) / tot) : 0.0f;
            ImGui::ProgressBar(frac);
            ImGui::SameLine();
            ImGui::Text("%s", status_message_.c_str());
        }
    }
    ImGui::End();
}

void AppState::render_quick_progress_view() {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("QuickProgressView", nullptr, flags)) {
        ImGui::Text("%s", status_message_.c_str());
        ImGui::Spacing();

        std::size_t done = completed_count_.load();
        std::size_t total = total_count_.load();
        float fraction = (total > 0) ? (static_cast<float>(done) / total) : 0.0f;

        char buf[128];
        std::snprintf(buf, sizeof(buf), "%zu / %zu (%.1f MB/s)", done, total, mb_per_sec_);
        ImGui::ProgressBar(fraction, ImVec2(-1, 28), buf);

        ImGui::Spacing();
        if (is_working_) {
            if (ImGui::Button("Cancel", ImVec2(100, 30))) {
                cancel_operation();
                close_requested_ = true;
            }
        } else {
            if (operation_finished_.load()) {
                auto now = std::chrono::steady_clock::now();
                double finish_elapsed = std::chrono::duration<double>(now - finish_time_).count();
                if (finish_elapsed > 1.2) {
                    close_requested_ = true;
                }
            }
            if (ImGui::Button("Close", ImVec2(100, 30))) {
                close_requested_ = true;
            }
        }
    }
    ImGui::End();
}

void AppState::render_context_menu_settings() {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Explorer / File Manager Integration:");
    ImGui::SameLine();

    char exe_buf[1024] = {0};
#if defined(_WIN32)
    GetModuleFileNameA(nullptr, exe_buf, sizeof(exe_buf));
#else
    std::string p = fs::canonical("/proc/self/exe").string();
    std::snprintf(exe_buf, sizeof(exe_buf), "%s", p.c_str());
#endif

    if (ctx_status_ == ContextMenuStatus::Registered) {
        ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "Registered (Active)");
        ImGui::SameLine();
        if (ImGui::Button("Remove Context Menu")) {
            if (unregister_context_menu()) {
                ctx_status_ = ContextMenuStatus::NotRegistered;
                ctx_feedback_ = "Removed context menu entries.";
            }
        }
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Not Registered");
        ImGui::SameLine();
        if (ImGui::Button("Register in Context Menu")) {
            if (register_context_menu(exe_buf)) {
                ctx_status_ = ContextMenuStatus::Registered;
                ctx_feedback_ = "Successfully registered context menu entries!";
            }
        }
    }

    if (!ctx_feedback_.empty()) {
        ImGui::SameLine();
        ImGui::Text("%s", ctx_feedback_.c_str());
    }
}

}  // namespace fzip::gui
