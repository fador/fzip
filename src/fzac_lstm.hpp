// fzip — Fast Multi-Layer Online Byte-Level Coupled-Gate LSTM Neural Predictor
// (bit-exact AVX2 + scalar fallback, zero libm calls in hot loop) for FZAC v3.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fzip::fzac_v3 {

class LstmModel {
  public:
    // `num_layers` in {1, 2}, `hidden_size` multiple of 8 (e.g. 64),
    // `horizon` is the truncated BPTT window length (e.g. 32).
    LstmModel(std::size_t num_layers = 1u, std::size_t hidden_size = 64u,
              std::size_t horizon = 32u);

    void start_byte() {}

    // Return exact conditional probability P(bit == 1 | prefix `c0`) in [1, 4095].
    auto predict_bit(std::uint32_t c0) const -> std::uint32_t;

    // Observe the actual byte, update output layer immediately + periodic BPTT,
    // and advance forward to predict the next byte.
    void update_byte(std::uint8_t actual_byte);

  private:
    struct Layer {
        std::size_t in_dim = 0;   // 0 for layer 0 (embedding only), H for layer 1 (h0 + embedding)
        // Input embedding [256 * H] for f, g, o
        std::vector<float> ef, eg, eo;
        std::vector<float> v_ef, v_eg, v_eo;
        // Inter-layer weights [H * in_dim] (only for layer >= 1)
        std::vector<float> uf, ug, uo;
        std::vector<float> v_uf, v_ug, v_uo;
        std::vector<float> g_uf, g_ug, g_uo;
        // Recurrent weights [H * H]
        std::vector<float> wf, wg, wo;
        std::vector<float> v_wf, v_wg, v_wo;
        std::vector<float> g_wf, g_wg, g_wo;
        // Biases [H]
        std::vector<float> bf, bg, bo;
        std::vector<float> v_bf, v_bg, v_bo;
        std::vector<float> g_bf, g_bg, g_bo;
        // Current state [H]
        std::vector<float> h_cur, c_cur;
    };

    struct LayerStep {
        std::vector<float> h_prev, c_prev;
        std::vector<float> f, g, o, tanh_c;
        std::vector<float> h_out;   // layer output h_t
        std::vector<float> dh_ext;  // gradient arriving from output / upper layer
    };

    struct StepCache {
        std::uint8_t in_byte = 0;
        std::vector<LayerStep> ls;
    };

    void forward_step(std::uint8_t in_byte);
    void run_bptt();
    void build_tree();

    std::size_t num_layers_;
    std::size_t H_;
    std::size_t total_H_;  // num_layers * H (concatenated readout!)
    std::size_t horizon_;
    std::size_t stride_;
    std::size_t step_ = 0;

    std::vector<Layer> layers_;

    // Concatenated readout: logits[256] = W_out[256 * total_H] * h_concat + b_out[256]
    std::vector<float> wout_;
    std::vector<float> v_wout_;
    std::vector<float> bout_;
    std::vector<float> v_bout_;
    std::vector<float> h_concat_;
    std::array<float, 256> probs_{};

    std::vector<StepCache> hist_;
    std::size_t hist_pos_ = 0;
    std::size_t hist_filled_ = 0;

    // Scratch buffers
    std::vector<float> dh_all_;
    std::vector<float> dh_next_;
    std::vector<float> dc_next_;
    std::vector<float> df_, dg_, do_;
    std::vector<float> dh_below_;

    std::array<std::uint32_t, 512> tree_{};
    bool avx2_ = false;
};

}  // namespace fzip::fzac_v3
