// fzip — Codec interface and per-file selector. Stage 5.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace fzip {

enum class CodecId : std::uint16_t {
    Store = 0,
    Deflate = 8,
    Zstd = 93,
};

// Result of compressing one entry.
struct CompressedEntry {
    CodecId codec{CodecId::Store};
    std::vector<std::byte> data;
};

namespace detail {

inline thread_local unsigned g_thread_budget = 0;  // 0 = auto

inline auto effective_threads() -> unsigned {
    if (g_thread_budget > 0) return g_thread_budget;
    unsigned hw = std::thread::hardware_concurrency();
    return hw ? hw : 1;
}

struct ThreadBudgetGuard {
    unsigned prev;
    explicit ThreadBudgetGuard(unsigned budget) : prev(g_thread_budget) {
        g_thread_budget = budget;
    }
    ~ThreadBudgetGuard() { g_thread_budget = prev; }
    ThreadBudgetGuard(const ThreadBudgetGuard&) = delete;
    auto operator=(const ThreadBudgetGuard&) -> ThreadBudgetGuard& = delete;
};

// Internal compression helpers that return empty `data` when Store is selected
// so `zip_writer` can stream directly from `entry.data` without copying.
auto compress_for_writer(CodecId codec, std::span<const std::byte> data,
                         int level, std::string_view hint_path) -> CompressedEntry;
auto compress_auto_for_writer(std::span<const std::byte> data,
                              std::string_view hint_path, int level)
    -> CompressedEntry;
auto decompress_span(CodecId codec, std::span<const std::byte> data,
                     std::size_t expected_size) -> std::vector<std::byte>;

}  // namespace detail

// Compress `data` using the selected `codec` and `level` (codec-specific).
// Returns an entry whose `codec` may differ from the request if Store is
// a better choice (e.g. compression failed to shrink the input).
auto compress(CodecId codec, std::span<const std::byte> data, int level,
              std::string_view hint_path) -> CompressedEntry;

// Auto-select a codec for `data` based on type detection, then trial-compress
// the promising candidates and keep the smallest result. `level` selects the
// strongest zstd level to try (>= 22 also enables the exhaustive parser).
auto compress_auto(std::span<const std::byte> data,
                   std::string_view hint_path, int level = 19)
    -> CompressedEntry;

// Decompress an entry back to its original bytes.
auto decompress(const CompressedEntry& entry) -> std::vector<std::byte>;

}  // namespace fzip
