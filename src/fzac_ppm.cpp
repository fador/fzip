// fzip — Variable-Order Byte-Level PPMd-Style Model with Exclusion, SEE, and
// Shkarin Text-Pointer Suffixes for FZAC v3 Context Mixing.
#include "fzac_ppm.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace fzip::fzac_v3 {

PpmModel::PpmModel(const std::uint8_t* buf, std::size_t mem_bytes, int max_order)
    : buf_(buf),
      max_order_(std::clamp(max_order, 2, 16)) {
    // Split arena roughly 55% PpmCtx nodes (16 B) + 45% PpmState entries (8 B).
    const std::size_t safe_bytes = std::max<std::size_t>(mem_bytes, 2u << 20);
    max_nodes_ = std::min<std::size_t>(
        (safe_bytes * 55u / 100u) / sizeof(PpmCtx), 0x3FFFFFF0u);
    max_states_ = std::min<std::size_t>(
        (safe_bytes * 45u / 100u) / sizeof(PpmState), 0x7FFFFFF0u);

    nodes_.reserve(std::min<std::size_t>(max_nodes_, 1u << 18));
    states_.reserve(std::min<std::size_t>(max_states_, 1u << 19));
    reset();
}

void PpmModel::reset() {
    nodes_.clear();
    states_.clear();
    for (auto& fl : free_lists_) fl.clear();

    // Node 0 = Order-0 root context
    PpmCtx root{};
    root.suffix = 0u;
    root.order = 0u;
    nodes_.push_back(root);

    max_ctx_ = 0u;
    text_parent_ = 0u;
    text_pos_ = 0u;
    text_edge_sym_ = 0u;
    text_active_ = false;
    last_max_order_ = 0;
    has_high_tree_ = false;
    num_visited_ = 0u;

    // Initialize Binary SEE table with calibrated Method-D priors
    for (int ord = 0; ord < 16; ++ord) {
        for (int fb = 0; fb < 8; ++fb) {
            const double freq_est = static_cast<double>(fb + 1);
            const double ord_bonus = 0.25 * static_cast<double>(std::min(ord, 8));
            const double p_hit = (freq_est + ord_bonus) / (freq_est + ord_bonus + 1.15);
            const std::uint16_t p12 = static_cast<std::uint16_t>(
                std::clamp(std::llround(p_hit * 4096.0), 64LL, 4032LL));
            for (int sb = 0; sb < 16; ++sb) {
                bin_see_[static_cast<std::size_t>(((ord << 3) | fb) * 16 + sb)] = p12;
            }
        }
    }

    // Initialize Multi-Symbol SEE table
    for (int ns = 0; ns < 16; ++ns) {
        for (int sd = 0; sd < 16; ++sd) {
            const std::uint32_t init_esc = static_cast<std::uint32_t>(
                std::clamp(4 + ns * 3 - sd / 2, 2, 48));
            multi_see_[static_cast<std::size_t>(ns * 16 + sd)].init(init_esc, 3u);
        }
    }
}

auto PpmModel::alloc_node(std::uint32_t suffix, std::uint8_t order) -> std::uint32_t {
    const std::uint32_t id = static_cast<std::uint32_t>(nodes_.size());
    PpmCtx node{};
    node.suffix = suffix;
    node.order = order;
    nodes_.push_back(node);
    return id;
}

auto PpmModel::alloc_states(std::uint8_t cap_log2) -> std::uint32_t {
    auto& fl = free_lists_[cap_log2];
    if (!fl.empty()) {
        const std::uint32_t off = fl.back();
        fl.pop_back();
        return off;
    }
    const std::uint32_t off = static_cast<std::uint32_t>(states_.size());
    const std::size_t cnt = std::size_t{1} << cap_log2;
    states_.resize(states_.size() + cnt);
    return off;
}

auto PpmModel::find_symbol(std::uint32_t ctx_idx, std::uint8_t sym,
                           std::uint32_t& out_succ) const -> bool {
    const auto& ctx = nodes_[ctx_idx];
    if (ctx.num_stats == 0u) return false;
    if (ctx.num_stats == 1u) {
        if (ctx.one_sym == sym) {
            out_succ = ctx.stats_idx;
            return true;
        }
        return false;
    }
    const PpmState* st = states_.data() + ctx.stats_idx;
    const std::uint16_t n = ctx.num_stats;
    for (std::uint16_t i = 0; i < n; ++i) {
        if (st[i].sym == sym) {
            out_succ = st[i].succ;
            return true;
        }
    }
    return false;
}

void PpmModel::rescale_ctx(std::uint32_t ctx_idx) {
    auto& ctx = nodes_[ctx_idx];
    if (ctx.num_stats == 1u) {
        ctx.sum_freq = static_cast<std::uint16_t>((ctx.sum_freq + 1u) >> 1);
        return;
    }
    PpmState* st = states_.data() + ctx.stats_idx;
    std::uint32_t new_sum = 0u;
    for (std::uint16_t i = 0; i < ctx.num_stats; ++i) {
        st[i].freq = static_cast<std::uint16_t>((st[i].freq + 1u) >> 1);
        new_sum += st[i].freq;
    }
    ctx.sum_freq = static_cast<std::uint16_t>(new_sum);
}

void PpmModel::add_or_inc_symbol(std::uint32_t ctx_idx, std::uint8_t sym,
                                 std::uint32_t succ, bool is_new,
                                 std::uint16_t add_freq) {
    if (!is_new) {
        auto& ctx = nodes_[ctx_idx];
        const std::uint32_t freq_lim = (ctx.order == 0u) ? kMaxFreqO0 : kMaxFreq;
        if (ctx.num_stats == 1u) {
            ctx.sum_freq = static_cast<std::uint16_t>(ctx.sum_freq + add_freq);
            if (succ != 0u) ctx.stats_idx = succ;
            if (ctx.sum_freq >= freq_lim) rescale_ctx(ctx_idx);
            return;
        }
        PpmState* st = states_.data() + ctx.stats_idx;
        for (std::uint16_t i = 0; i < ctx.num_stats; ++i) {
            if (st[i].sym == sym) {
                st[i].freq = static_cast<std::uint16_t>(st[i].freq + add_freq);
                if (succ != 0u) st[i].succ = succ;
                ctx.sum_freq = static_cast<std::uint16_t>(ctx.sum_freq + add_freq);
                // Keep higher-frequency symbols near front (faster linear scan)
                if (i > 0u && st[i].freq > st[i - 1u].freq) {
                    std::swap(st[i], st[i - 1u]);
                }
                if (ctx.sum_freq >= freq_lim) rescale_ctx(ctx_idx);
                return;
            }
        }
        return;
    }

    // Adding a brand-new symbol to ctx_idx
    if (nodes_[ctx_idx].num_stats == 0u) {
        auto& ctx = nodes_[ctx_idx];
        ctx.num_stats = 1u;
        ctx.one_sym = sym;
        ctx.sum_freq = add_freq;
        ctx.stats_idx = succ;
        return;
    }

    if (nodes_[ctx_idx].num_stats == 1u) {
        const std::uint8_t old_sym = nodes_[ctx_idx].one_sym;
        const std::uint16_t old_freq = nodes_[ctx_idx].sum_freq;
        const std::uint32_t old_succ = nodes_[ctx_idx].stats_idx;
        const std::uint32_t off = alloc_states(1u);  // capacity 2
        PpmState* st = states_.data() + off;
        st[0] = {old_succ, old_freq, old_sym, 0u};
        st[1] = {succ, add_freq, sym, 0u};
        auto& ctx = nodes_[ctx_idx];
        ctx.stats_idx = off;
        ctx.cap_log2 = 1u;
        ctx.num_stats = 2u;
        ctx.sum_freq = static_cast<std::uint16_t>(old_freq + add_freq);
        return;
    }

    const std::uint16_t cur_n = nodes_[ctx_idx].num_stats;
    if (cur_n >= 256u) return;
    const std::uint8_t cap_l2 = nodes_[ctx_idx].cap_log2;
    const std::uint16_t cap = static_cast<std::uint16_t>(1u << cap_l2);
    if (cur_n >= cap) {
        const std::uint8_t new_l2 = static_cast<std::uint8_t>(cap_l2 + 1u);
        const std::uint32_t old_off = nodes_[ctx_idx].stats_idx;
        const std::uint32_t new_off = alloc_states(new_l2);
        std::memcpy(states_.data() + new_off, states_.data() + old_off,
                    static_cast<std::size_t>(cur_n) * sizeof(PpmState));
        free_lists_[cap_l2].push_back(old_off);
        nodes_[ctx_idx].stats_idx = new_off;
        nodes_[ctx_idx].cap_log2 = new_l2;
    }
    auto& ctx = nodes_[ctx_idx];
    states_[ctx.stats_idx + cur_n] = {succ, add_freq, sym, 0u};
    ctx.num_stats = static_cast<std::uint16_t>(cur_n + 1u);
    ctx.sum_freq = static_cast<std::uint16_t>(ctx.sum_freq + add_freq);
    const std::uint32_t freq_lim = (ctx.order == 0u) ? kMaxFreqO0 : kMaxFreq;
    if (ctx.sum_freq >= freq_lim) rescale_ctx(ctx_idx);
}

auto PpmModel::materialize_text_chain(std::uint32_t parent_idx, std::uint8_t edge_sym,
                                      std::uint32_t text_pos, std::size_t cur_pos)
    -> std::uint32_t {
    // Check if edge_sym in parent_idx already points to a materialized node
    std::uint32_t existing_succ = 0u;
    const bool found = find_symbol(parent_idx, edge_sym, existing_succ);
    if (found && existing_succ != 0u && (existing_succ & kTextFlag) == 0u) {
        return existing_succ;
    }

    const std::uint8_t parent_ord = nodes_[parent_idx].order;
    std::uint32_t suf_child = 0u;
    if (parent_ord > 0u) {
        const std::uint32_t parent_suf = nodes_[parent_idx].suffix;
        suf_child = materialize_text_chain(parent_suf, edge_sym, text_pos, cur_pos);
    }

    // Re-check after recursion (vector may have reallocated)
    const std::uint32_t child_id = alloc_node(suf_child, static_cast<std::uint8_t>(parent_ord + 1u));

    // Seed child with the single byte observed at `text_pos` if valid (< cur_pos)
    const std::uint32_t tp = (found && (existing_succ & kTextFlag) != 0u)
                                 ? (existing_succ & ~kTextFlag)
                                 : text_pos;
    if (tp < cur_pos) {
        auto& child = nodes_[child_id];
        child.num_stats = 1u;
        child.one_sym = buf_[tp];
        child.sum_freq = 1u;
        child.stats_idx = kTextFlag | (tp + 1u);
    }

    // Point parent's edge_sym to `child_id`
    if (found) {
        add_or_inc_symbol(parent_idx, edge_sym, child_id, /*is_new=*/false, 0u);
    } else {
        add_or_inc_symbol(parent_idx, edge_sym, child_id, /*is_new=*/true, 1u);
    }
    return child_id;
}

void PpmModel::start_byte(std::size_t pos) {
    std::array<double, 256> probs{};
    std::array<double, 256> high_probs{};
    std::array<std::uint8_t, 256> excl{};
    std::size_t excl_cnt = 0u;
    double remaining = 16777216.0;

    num_visited_ = 0u;
    last_max_order_ = 0;
    has_high_tree_ = false;

    // 1. Check active text-pointer prediction (Order = parent.order + 1)
    if (text_active_ && text_pos_ < pos) {
        const std::uint8_t sym = buf_[text_pos_];
        const int ord = std::min<int>(nodes_[text_parent_].order + 1, max_order_);
        last_max_order_ = ord;

        // Measure backward match length beyond parent.order for sharper confidence
        int match_ext = ord;
        const std::size_t max_check = std::min<std::size_t>(text_pos_, 16u);
        while (match_ext < static_cast<int>(max_check) &&
               buf_[text_pos_ - 1u - static_cast<std::size_t>(match_ext)] ==
                   buf_[pos - 1u - static_cast<std::size_t>(match_ext)]) {
            ++match_ext;
        }
        const int ord_b = std::min(match_ext, 15);
        const std::size_t bidx = static_cast<std::size_t>(((ord_b << 3) | 0) * 16);
        const double p_sym = static_cast<double>(bin_see_[bidx]) / 4096.0;
        probs[sym] += remaining * p_sym;
        const double h_unif = (16777216.0 * (1.0 - p_sym)) / 255.0;
        high_probs.fill(h_unif);
        high_probs[sym] = p_sym * 16777216.0;
        has_high_tree_ = true;
        excl[sym] = 1u;
        excl_cnt = 1u;
        remaining *= (1.0 - p_sym);
        if (num_visited_ < visited_.size()) {
            visited_[num_visited_++] = {0xFFFFFFFFu, bidx, true};
        }
    }

    // 2. Walk materialized suffix chain from max_ctx_ down to root (Order 0)
    std::uint32_t ctx_idx = max_ctx_;
    for (;;) {
        const auto& ctx = nodes_[ctx_idx];
        const std::uint16_t ns = ctx.num_stats;
        if (ns == 1u) {
            const std::uint8_t sym = ctx.one_sym;
            if (!excl[sym]) {
                if (last_max_order_ == 0) last_max_order_ = ctx.order;
                const int ord_b = std::min<int>(ctx.order, 15);
                const int freq_b = std::min<int>(std::bit_width(ctx.sum_freq), 7);
                const int suf_b = (ctx_idx != 0u)
                                      ? std::min<int>(nodes_[ctx.suffix].num_stats >> 1, 15)
                                      : 0;
                const std::size_t bidx =
                    static_cast<std::size_t>(((ord_b << 3) | freq_b) * 16 + suf_b);
                const double p_sym = static_cast<double>(bin_see_[bidx]) / 4096.0;
                probs[sym] += remaining * p_sym;
                if (!has_high_tree_) {
                    const double h_unif = (16777216.0 * (1.0 - p_sym)) / 255.0;
                    high_probs.fill(h_unif);
                    high_probs[sym] = p_sym * 16777216.0;
                    has_high_tree_ = true;
                }
                excl[sym] = 1u;
                ++excl_cnt;
                remaining *= (1.0 - p_sym);
                if (num_visited_ < visited_.size()) {
                    visited_[num_visited_++] = {ctx_idx, bidx, true};
                }
            }
        } else if (ns >= 2u) {
            const PpmState* st = states_.data() + ctx.stats_idx;
            std::uint32_t unexcl_sum = 0u;
            std::uint32_t unexcl_cnt = 0u;
            for (std::uint16_t i = 0; i < ns; ++i) {
                if (!excl[st[i].sym]) {
                    unexcl_sum += st[i].freq;
                    ++unexcl_cnt;
                }
            }
            if (unexcl_cnt > 0u) {
                if (last_max_order_ == 0) last_max_order_ = ctx.order;
                const int ns_b = std::min<int>(std::bit_width(unexcl_cnt), 15);
                const int sd_b = std::min<int>(
                    std::bit_width(static_cast<std::uint32_t>(
                        unexcl_sum > unexcl_cnt ? (unexcl_sum - unexcl_cnt) : 1u)),
                    15);
                const std::size_t sidx = static_cast<std::size_t>(ns_b * 16 + sd_b);
                const double esc_w =
                    static_cast<double>(multi_see_[sidx].mean()) * 0.25 +
                    static_cast<double>(unexcl_cnt) * 0.5;
                const double p_esc = std::clamp(
                    esc_w / (static_cast<double>(unexcl_sum) + esc_w), 0.002, 0.98);
                const double scale =
                    remaining * (1.0 - p_esc) / static_cast<double>(unexcl_sum);
                const bool fill_high = !has_high_tree_;
                const double hscale =
                    fill_high ? (16777216.0 * (1.0 - p_esc) / static_cast<double>(unexcl_sum)) : 0.0;
                if (fill_high) {
                    const double h_unif =
                        (16777216.0 * p_esc) / static_cast<double>(std::max<std::size_t>(1u, 256u - unexcl_cnt));
                    high_probs.fill(h_unif);
                }
                for (std::uint16_t i = 0; i < ns; ++i) {
                    const std::uint8_t s = st[i].sym;
                    if (!excl[s]) {
                        probs[s] += static_cast<double>(st[i].freq) * scale;
                        if (fill_high) {
                            high_probs[s] = static_cast<double>(st[i].freq) * hscale;
                        }
                        excl[s] = 1u;
                        ++excl_cnt;
                    }
                }
                if (fill_high) has_high_tree_ = true;
                remaining *= p_esc;
                if (num_visited_ < visited_.size()) {
                    visited_[num_visited_++] = {ctx_idx, sidx, false};
                }
            }
        }

        if (ctx_idx == 0u) break;
        ctx_idx = ctx.suffix;
    }

    // 3. Order -1 uniform fallback for unexcluded symbols
    if (excl_cnt < 256u) {
        const double unif = remaining / static_cast<double>(256u - excl_cnt);
        for (std::size_t b = 0; b < 256u; ++b) {
            if (!excl[b]) probs[b] += unif;
        }
    }

    // 4. Quantize to uint32_t and build bottom-up binary sum tree
    for (std::size_t b = 0; b < 256u; ++b) {
        const std::uint32_t v = static_cast<std::uint32_t>(probs[b]);
        tree_[256u + b] = (v == 0u) ? 1u : v;
    }
    for (std::size_t i = 255u; i >= 1u; --i) {
        tree_[i] = tree_[i * 2u] + tree_[i * 2u + 1u];
    }

    if (has_high_tree_) {
        for (std::size_t b = 0; b < 256u; ++b) {
            const std::uint32_t v = static_cast<std::uint32_t>(high_probs[b]);
            high_tree_[256u + b] = (v == 0u) ? 1u : v;
        }
        for (std::size_t i = 255u; i >= 1u; --i) {
            high_tree_[i] = high_tree_[i * 2u] + high_tree_[i * 2u + 1u];
        }
    }
}

auto PpmModel::predict_bit(std::uint32_t c0) const -> std::uint32_t {
    const std::uint32_t tot = tree_[c0];
    const std::uint32_t p1 = tree_[c0 * 2u + 1u];
    const std::uint64_t p = (static_cast<std::uint64_t>(p1) * 4096u + (tot >> 1)) / tot;
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(p, 1u, 4095u));
}

auto PpmModel::predict_high_order_bit(std::uint32_t c0) const -> std::uint32_t {
    if (!has_high_tree_) return 2048u;
    const std::uint32_t tot = high_tree_[c0];
    const std::uint32_t p1 = high_tree_[c0 * 2u + 1u];
    const std::uint64_t p = (static_cast<std::uint64_t>(p1) * 4096u + (tot >> 1)) / tot;
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(p, 1u, 4095u));
}

void PpmModel::update_byte(std::size_t pos, std::uint8_t byte) {
    // 1. Update SEE tables for visited contexts
    for (std::size_t i = 0; i < num_visited_; ++i) {
        const auto& v = visited_[i];
        bool hit = false;
        if (v.ctx_idx == 0xFFFFFFFFu) {
            hit = (text_active_ && text_pos_ < pos && buf_[text_pos_] == byte);
        } else {
            std::uint32_t dummy = 0u;
            hit = find_symbol(v.ctx_idx, byte, dummy);
        }

        if (v.is_bin) {
            std::int32_t p = bin_see_[v.see_idx];
            if (hit) {
                p += (4096 - p) >> 5;
            } else {
                p -= p >> 5;
            }
            bin_see_[v.see_idx] = static_cast<std::uint16_t>(std::clamp(p, 32, 4064));
        } else {
            multi_see_[v.see_idx].update(!hit);
        }
        if (hit) break;
    }

    // Check arena headroom before allocating nodes/states
    if (nodes_.size() + 32u >= max_nodes_ || states_.size() + 512u >= max_states_) {
        reset();
        // Still record `byte` in root so Order-0 stays warm
        add_or_inc_symbol(0u, byte, kTextFlag | static_cast<std::uint32_t>(pos + 1u),
                          /*is_new=*/true, 1u);
        return;
    }

    // 2. Materialize active text-pointer chain if valid
    if (text_active_ && text_pos_ < pos) {
        if (nodes_[text_parent_].order < static_cast<std::uint8_t>(max_order_)) {
            max_ctx_ = materialize_text_chain(text_parent_, text_edge_sym_, text_pos_, pos);
        }
        text_active_ = false;
    }

    // 3. Walk suffix chain with Update Exclusion
    std::array<std::uint32_t, 20> missed{};
    std::size_t num_missed = 0u;
    std::uint32_t ctx_idx = max_ctx_;
    std::uint32_t found_ctx = 0xFFFFFFFFu;
    std::uint32_t found_succ = 0u;

    for (;;) {
        std::uint32_t succ = 0u;
        if (find_symbol(ctx_idx, byte, succ)) {
            found_ctx = ctx_idx;
            found_succ = succ;
            break;
        }
        if (num_missed < missed.size()) {
            missed[num_missed++] = ctx_idx;
        }
        if (ctx_idx == 0u) break;
        ctx_idx = nodes_[ctx_idx].suffix;
    }

    const std::uint32_t new_text_succ =
        (pos + 1u < 0x7FFFFFFEu) ? (kTextFlag | static_cast<std::uint32_t>(pos + 1u)) : 0u;

    // 4. Add `byte` to all missed contexts (attaching cheap text pointer!)
    for (std::size_t i = 0; i < num_missed; ++i) {
        const std::uint32_t m_ctx = missed[i];
        const std::uint32_t child_succ =
            (nodes_[m_ctx].order < static_cast<std::uint8_t>(max_order_)) ? new_text_succ : 0u;
        add_or_inc_symbol(m_ctx, byte, child_succ, /*is_new=*/true, 1u);
    }

    // Helper: find deepest materialized suffix context after appending `byte`
    auto find_materialized_suffix = [&](std::uint32_t start_ctx) -> std::uint32_t {
        std::uint32_t c = start_ctx;
        for (;;) {
            if (c == 0u) {
                std::uint32_t rs = 0u;
                if (find_symbol(0u, byte, rs) && rs != 0u && (rs & kTextFlag) == 0u) {
                    return rs;
                }
                return 0u;
            }
            c = nodes_[c].suffix;
            std::uint32_t ss = 0u;
            if (find_symbol(c, byte, ss) && ss != 0u && (ss & kTextFlag) == 0u) {
                return ss;
            }
        }
    };

    // 5. Update `found_ctx` and advance active context for next byte
    if (found_ctx != 0xFFFFFFFFu) {
        const std::uint16_t inc = (nodes_[found_ctx].order > 0u) ? 2u : 1u;
        const std::uint32_t upd_succ =
            (found_succ == 0u && nodes_[found_ctx].order < static_cast<std::uint8_t>(max_order_))
                ? new_text_succ
                : 0u;
        add_or_inc_symbol(found_ctx, byte, upd_succ, /*is_new=*/false, inc);

        if (found_succ != 0u && (found_succ & kTextFlag) == 0u) {
            // Already materialized child node!
            max_ctx_ = found_succ;
            text_active_ = false;
        } else if (found_succ != 0u && (found_succ & kTextFlag) != 0u &&
                   nodes_[found_ctx].order < static_cast<std::uint8_t>(max_order_)) {
            // Active text-pointer match! Set max_ctx_ to the deepest materialized
            // suffix context AFTER `byte` (NOT `found_ctx`, which is BEFORE `byte`!).
            max_ctx_ = find_materialized_suffix(found_ctx);
            text_parent_ = found_ctx;
            text_pos_ = found_succ & ~kTextFlag;
            text_edge_sym_ = byte;
            text_active_ = true;
        } else {
            // Reached max_order: advance along suffix of `found_ctx` on `byte`
            if (found_ctx != 0u) {
                const std::uint32_t suf = nodes_[found_ctx].suffix;
                std::uint32_t suf_succ = 0u;
                if (find_symbol(suf, byte, suf_succ) && suf_succ != 0u) {
                    if ((suf_succ & kTextFlag) == 0u) {
                        max_ctx_ = suf_succ;
                        text_active_ = false;
                    } else {
                        max_ctx_ = find_materialized_suffix(suf);
                        text_parent_ = suf;
                        text_pos_ = suf_succ & ~kTextFlag;
                        text_edge_sym_ = byte;
                        text_active_ = true;
                    }
                } else {
                    max_ctx_ = find_materialized_suffix(suf);
                    text_active_ = false;
                }
            } else {
                max_ctx_ = 0u;
                text_active_ = false;
            }
        }
    } else {
        max_ctx_ = 0u;
        text_active_ = false;
    }
}

}  // namespace fzip::fzac_v3
