#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "codec.hpp"
#include "zip_reader.hpp"
#include "zip_writer.hpp"
#include "context_menu.hpp"

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

private:
    void render_compress_view();
    void render_inspect_view();
    void render_quick_progress_view();
    void render_context_menu_settings();

    void start_compression();
    void cancel_operation();
    void open_archive(const std::string& path);
    void extract_archive(const std::string& out_dir);

public:
    void add_input_paths(const std::vector<std::string>& paths);

private:
    AppMode mode_ = AppMode::Compress;
    bool close_requested_ = false;

    // Compression inputs
    std::vector<FileItem> input_items_;
    std::string output_archive_path_;
    int selected_codec_idx_ = 0; // 0 = Auto, 1 = FZAC, 2 = Zstd, 3 = Deflate, 4 = Store
    int compression_level_ = 19;

    // Live compression / progress state
    std::atomic<bool> is_working_{false};
    std::atomic<bool> operation_finished_{false};
    std::chrono::steady_clock::time_point finish_time_{};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<std::size_t> completed_count_{0};
    std::atomic<std::size_t> total_count_{0};
    std::atomic<std::uint64_t> total_uncompressed_bytes_{0};
    std::atomic<std::uint64_t> total_compressed_bytes_{0};

    std::string status_message_;
    std::chrono::steady_clock::time_point start_time_;
    double elapsed_seconds_ = 0.0;
    double mb_per_sec_ = 0.0;

    std::mutex rows_mutex_;
    std::vector<CompressionRow> rows_;

    std::thread worker_thread_;

    // Archive inspection
    std::string inspected_archive_path_;
    std::vector<ZipReadEntry> inspected_entries_;
    std::uint64_t inspected_total_uncomp_ = 0;
    std::uint64_t inspected_total_comp_ = 0;

    // Context menu status
    ContextMenuStatus ctx_status_ = ContextMenuStatus::Unknown;
    std::string ctx_feedback_;
};

}  // namespace fzip::gui
