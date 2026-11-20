#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "codec.hpp"
#include "context_menu.hpp"
#include "zip_reader.hpp"
#include "zip_writer.hpp"

struct ImFont;

namespace fzip::gui {

enum class AppMode {
    Compress,
    Inspect,
    QuickProgress
};

struct FileItem {
    std::string path;
    std::uint64_t size = 0;
    bool is_directory = false;
};

struct CompressionRow {
    std::string name;
    std::uint64_t uncompressed_size = 0;
    std::uint64_t compressed_size = 0;
    CodecId codec = CodecId::Store;
    bool completed = false;
};

class AppState {
public:
    AppState();
    ~AppState();

    void init(int argc, char** argv);
    void render();

    bool should_close() const { return close_requested_; }
    bool is_quick_mode() const { return mode_ == AppMode::QuickProgress; }
    void set_fonts(ImFont* bold_font, float ui_scale) {
        bold_font_ = bold_font;
        ui_scale_ = ui_scale;
    }
    void set_native_window(void* hwnd) { native_hwnd_ = hwnd; }

private:
    void render_header_bar();
    void render_compress_view();
    void render_inspect_view();
    void render_quick_progress_view();
    void render_context_menu_controls();

    void start_compression();
    void cancel_operation();
    void open_archive(const std::string& path);
    void extract_archive(const std::string& out_dir);
    void set_status(std::string msg);
    auto get_status() -> std::string;

    auto try_join_or_start_ipc(const char* pipe_id,
                               const std::vector<std::string>& paths) -> bool;
    void stop_ipc_server();
    void drain_ipc_paths();

public:
    void add_input_paths(const std::vector<std::string>& paths);

private:
    AppMode mode_ = AppMode::Compress;
    bool close_requested_ = false;
    ImFont* bold_font_ = nullptr;
    float ui_scale_ = 1.0f;
    void* native_hwnd_ = nullptr;

    // Compression inputs
    std::vector<FileItem> input_items_;
    std::string output_archive_path_;
    bool auto_output_path_ = true;
    int selected_codec_idx_ = 0;  // 0 = Auto, 1 = FZAC, 2 = Zstd, 3 = Deflate, 4 = Store
    int compression_level_ = 19;

    // Single-instance Explorer multi-select IPC
    std::mutex ipc_mutex_;
    std::vector<std::string> pending_ipc_paths_;
    std::chrono::steady_clock::time_point last_ipc_time_{};
    bool pending_quick_start_ = false;
    std::thread ipc_thread_;
    void* ipc_pipe_handle_ = nullptr;
    void* ipc_stop_event_ = nullptr;

    // Live compression / extraction progress state
    std::atomic<bool> is_working_{false};
    std::atomic<bool> is_extract_op_{false};
    std::atomic<bool> operation_finished_{false};
    std::atomic<bool> operation_failed_{false};
    std::chrono::steady_clock::time_point finish_time_{};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<std::size_t> completed_count_{0};
    std::atomic<std::size_t> total_count_{0};
    std::atomic<std::uint64_t> total_uncompressed_bytes_{0};
    std::atomic<std::uint64_t> total_compressed_bytes_{0};

    std::mutex status_mutex_;
    std::string status_message_;
    std::chrono::steady_clock::time_point start_time_;
    double elapsed_seconds_ = 0.0;
    double mb_per_sec_ = 0.0;

    std::mutex rows_mutex_;
    std::vector<CompressionRow> rows_;

    std::thread worker_thread_;

    // Archive inspection & extraction
    std::string inspected_archive_path_;
    std::string last_extracted_dir_;
    std::vector<ZipReadEntry> inspected_entries_;
    std::uint64_t inspected_total_uncomp_ = 0;
    std::uint64_t inspected_total_comp_ = 0;
    std::array<char, 256> inspect_filter_{};

    // Context menu status
    ContextMenuStatus ctx_status_ = ContextMenuStatus::Unknown;
    std::string ctx_feedback_;
};

}  // namespace fzip::gui
