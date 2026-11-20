// fzip — Variable-Order Byte-Level PPMd-Style Model with Exclusion, SEE, and
// Shkarin Text-Pointer Suffixes for FZAC v3 Context Mixing.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fzip::fzac_v3 {

class PpmModel {
  public:
    // `buf` points directly to the caller's raw/decoded byte buffer (no copy).
    // `mem_bytes` sets the arena cap for PPM nodes & state arrays.
    // `max_order` is the maximum context order (e.g. 8..16).
    PpmModel(const std::uint8_t* buf, std::size_t mem_bytes, int max_order);

    // Compute exact 256-byte probability distribution with exclusion at byte `pos`
    // and build the 255-node binary probability tree for O(1) bit queries.
    void start_byte(std::size_t pos);

    // Return exact conditional probability P(bit == 1 | prefix `c0`) in [1, 4095],
    // plus highest matched order (0..max_order) for mixer meta-features.
    auto predict_bit(std::uint32_t c0) const -> std::uint32_t;
    auto predict_high_order_bit(std::uint32_t c0) const -> std::uint32_t;
    auto matched_order() const -> int { return last_max_order_; }

    // Update the suffix trie and SEE tables after observing `byte` at `pos`.
    void update_byte(std::size_t pos, std::uint8_t byte);

  private:
    static constexpr std::uint32_t kTextFlag = 0x80000000u;
    static constexpr std::uint32_t kMaxFreqO0 = 512u;
    static constexpr std::uint32_t kMaxFreq   = 128u;

    struct PpmState {
        std::uint32_t succ = 0;   // 0 = none, <kTextFlag = node idx, >=kTextFlag = text pos
        std::uint16_t freq = 0;
        std::uint8_t sym = 0;
        std::uint8_t pad = 0;
    };

    struct PpmCtx {
        std::uint32_t suffix = 0;     // suffix context node index (0 = root)
        std::uint32_t stats_idx = 0;  // if num_stats==1: succ of one_sym; else: pool offset
        std::uint16_t sum_freq = 0;   // total freq (or freq of one_sym)
        std::uint16_t num_stats = 0;  // 0..256
        std::uint8_t one_sym = 0;     // inline symbol when num_stats == 1
        std::uint8_t order = 0;       // 0..max_order
        std::uint8_t cap_log2 = 0;    // state array capacity = 1 << cap_log2 (when num_stats >= 2)
        std::uint8_t flags = 0;       // bit 0: recent symbol repeated
    };

    struct SeeEntry {
        std::uint16_t summ = 0;   // shifted probability numerator
        std::uint8_t shift = 3;
        std::uint8_t count = 4;

        void init(std::uint32_t init_val, std::uint8_t sh = 3) {
            shift = sh;
            summ = static_cast<std::uint16_t>(init_val << sh);
            count = 4;
        }

        // Returns escape weight in [1, 255] (approx numerator over 256 or relative to sum).
        auto mean() const -> std::uint32_t {
            const std::uint32_t ret = summ >> shift;
            return (ret == 0u) ? 1u : ret;
        }

        void update(bool escaped) {
            if (escaped) {
                summ = static_cast<std::uint16_t>(std::min<std::uint32_t>(summ + 16u, 65000u));
            } else {
                const std::uint32_t dec = std::max<std::uint32_t>(1u, summ >> 6);
                summ = (summ > dec) ? static_cast<std::uint16_t>(summ - dec) : 1u;
            }
            if (--count == 0u) {
                if (shift < 7u) {
                    summ = static_cast<std::uint16_t>(summ << 1);
                    ++shift;
                }
                count = static_cast<std::uint8_t>(3u << shift);
            }
        }
    };

    void reset();
    auto alloc_node(std::uint32_t suffix, std::uint8_t order) -> std::uint32_t;
    auto alloc_states(std::uint8_t cap_log2) -> std::uint32_t;
    auto find_symbol(std::uint32_t ctx_idx, std::uint8_t sym,
                     std::uint32_t& out_succ) const -> bool;
    void add_or_inc_symbol(std::uint32_t ctx_idx, std::uint8_t sym,
                           std::uint32_t succ, bool is_new, std::uint16_t add_freq);
    void rescale_ctx(std::uint32_t ctx_idx);
    auto materialize_text_chain(std::uint32_t parent_idx, std::uint8_t edge_sym,
                                std::uint32_t text_pos, std::size_t cur_pos) -> std::uint32_t;

    const std::uint8_t* buf_;
    std::size_t max_nodes_;
    std::size_t max_states_;
    int max_order_;

    std::vector<PpmCtx> nodes_;
    std::vector<PpmState> states_;
    std::array<std::vector<std::uint32_t>, 9> free_lists_{};  // cap_log2 = 1..8

    // Active suffix chain at current byte boundary
    std::uint32_t max_ctx_ = 0;         // deepest materialized node
    std::uint32_t text_parent_ = 0;     // parent node if a text-pointer match is active
    std::uint32_t text_pos_ = 0;        // target index in buf_ (if text_active_)
    std::uint8_t text_edge_sym_ = 0;    // edge symbol from text_parent_ to text_pos_
    bool text_active_ = false;

    // Adaptive Binary SEE (for num_stats == 1) & Multi-Symbol SEE (for num_stats >= 2)
    std::array<std::uint16_t, 128 * 16> bin_see_{};  // 12-bit prob of hitting the 1 symbol
    std::array<SeeEntry, 16 * 16> multi_see_{};

    // Per-byte binary probability trees (256 leaves at [256..511], sums at [1..255])
    std::array<std::uint32_t, 512> tree_{};
    std::array<std::uint32_t, 512> high_tree_{};
    int last_max_order_ = 0;
    bool has_high_tree_ = false;

    // Scratch state for SEE update at update_byte
    struct VisitedLevel {
        std::uint32_t ctx_idx;
        std::size_t see_idx;
        bool is_bin;
    };
    std::array<VisitedLevel, 20> visited_{};
    std::size_t num_visited_ = 0;
};

}  // namespace fzip::fzac_v3
