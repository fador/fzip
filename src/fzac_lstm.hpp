// fzip — Upgraded Multi-Layer Online Byte-Level Coupled-Gate LSTM Neural
// Predictor with RMSNorm, Adam Momentum, and Sparse Bigram/WRT Token Embeddings
// (bit-exact AVX2 + scalar fallback) for FZAC v3.
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

    // 4-bit quantized neural hidden-state regime (0..15) for APM conditioning.
    auto state_hash() const -> std::uint32_t { return state_hash_; }

    // 2-bit prediction confidence / peakedness bucket (0..3) from current softmax.
    auto conf_bucket() const -> std::uint32_t { return conf_bucket_; }

    // Observe `actual_byte` (and optional hashed bigram / WRT token ID `tok_id`),
    // perform online Adam output update + periodic RMSNorm BPTT, and advance forward.
    void update_byte(std::uint8_t actual_byte, std::uint32_t tok_id = 0u);

  private:
    static constexpr std::size_t kTokBuckets = 2048u;
    static constexpr std::size_t kTokMask    = kTokBuckets - 1u;

    struct Layer {
        std::size_t in_dim = 0;  // 0 for layer 0, H for layer 1
        // 1-byte input embeddings [256 * H] for f, g, o
        std::vector<float> ef, eg, eo;
        std::vector<float> m_ef, m_eg, m_eo;
        std::vector<float> v_ef, v_eg, v_eo;
        // Sparse bigram / WRT token embeddings [kTokBuckets * H] (layer 0 only)
        std::vector<float> tf, tg, to;
        std::vector<float> v_tf, v_tg, v_to;
        // Inter-layer weights [H * in_dim] (layer >= 1)
        std::vector<float> uf, ug, uo;
        std::vector<float> m_uf, m_ug, m_uo;
        std::vector<float> v_uf, v_ug, v_uo;
        std::vector<float> g_uf, g_ug, g_uo;
        // Recurrent weights [H * H]
        std::vector<float> wf, wg, wo;
        std::vector<float> m_wf, m_wg, m_wo;
        std::vector<float> v_wf, v_wg, v_wo;
        std::vector<float> g_wf, g_wg, g_wo;
        // RMSNorm scale gains [H] and biases [H]
        std::vector<float> gf, gg, go;
        std::vector<float> m_gf, m_gg, m_go;
        std::vector<float> v_gf, v_gg, v_go;
        std::vector<float> g_gf, g_gg, g_go;

        std::vector<float> bf, bg, bo;
        std::vector<float> m_bf, m_bg, m_bo;
        std::vector<float> v_bf, v_bg, v_bo;
        std::vector<float> g_bf, g_bg, g_bo;
        // Current state [H]
        std::vector<float> h_cur, c_cur;
    };

    struct LayerStep {
        std::vector<float> h_prev, c_prev;
        // RMSNorm saved normalized pre-activations & inverse RMS for exact backprop
        std::vector<float> uf_n, ug_n, uo_n;
        float r_f = 1.0f, r_g = 1.0f, r_o = 1.0f;
        std::vector<float> f, g, o, tanh_c;
        std::vector<float> h_out;
        std::vector<float> dh_ext;
    };

    struct StepCache {
        std::uint8_t in_byte = 0;
        std::uint16_t tok_id = 0;
        std::vector<LayerStep> ls;
    };

    void forward_step(std::uint8_t in_byte, std::uint16_t tok_id);
    void run_bptt();
    void build_tree();

    std::size_t num_layers_;
    std::size_t H_;
    std::size_t total_H_;
    std::size_t horizon_;
    std::size_t stride_;
    std::size_t step_ = 0;

    std::vector<Layer> layers_;

    // Concatenated readout: logits[256] = W_out[256 * total_H] * h_concat + b_out[256]
    std::vector<float> wout_;
    std::vector<float> m_wout_;
    std::vector<float> v_wout_;
    std::vector<float> bout_;
    std::vector<float> m_bout_;
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
    std::vector<float> af_n_, ag_n_, ao_n_;
    std::vector<float> dh_below_;

    std::array<std::uint32_t, 512> tree_{};
    std::uint32_t state_hash_ = 0u;
    std::uint32_t conf_bucket_ = 0u;
    bool avx2_ = false;
};

}  // namespace fzip::fzac_v3
