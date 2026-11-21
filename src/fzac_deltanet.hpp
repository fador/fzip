// fzip — Online Byte-Level Gated DeltaNet (Fast-Weight Associative Matrix-Memory RNN)
// Bit-exact AVX2 + scalar fallback for FZAC v3.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fzip::fzac_v3 {

class DeltaNetModel {
  public:
    // `num_heads` (default 4), `d_head` multiple of 8 (default 32 -> 4096-float matrix state),
    // `d_in` multiple of 8 (default 64), `horizon` BPTT window (default 16).
    DeltaNetModel(std::size_t num_heads = 4u, std::size_t d_head = 32u,
                  std::size_t d_in = 64u, std::size_t horizon = 16u);

    void start_byte() {}

    // Return exact conditional probability P(bit == 1 | prefix `c0`) in [1, 4095].
    auto predict_bit(std::uint32_t c0) const -> std::uint32_t;

    // 2-bit prediction confidence / peakedness bucket (0..3) from current softmax.
    auto conf_bucket() const -> std::uint32_t { return conf_bucket_; }

    // Observe `actual_byte` (and hashed bigram/WRT token ID `tok_id`),
    // perform online Adam readout update + periodic exact Delta-Rule BPTT,
    // and advance associative matrix memory S_t forward.
    void update_byte(std::uint8_t actual_byte, std::uint32_t tok_id = 0u);

  private:
    static constexpr std::size_t kConvK      = 4u;
    static constexpr std::size_t kTokBuckets = 2048u;
    static constexpr std::size_t kTokMask    = kTokBuckets - 1u;

    struct StepCache {
        std::uint8_t in_byte = 0;
        std::uint16_t tok_id = 0;
        // 4-tap conv input history e_{t-0..3} [kConvK * d_in]
        std::vector<float> conv_e;
        std::vector<float> conv_u;   // pre-SiLU [d_in]
        std::vector<float> x_un;     // post-SiLU normalized [d_in]
        std::vector<float> x_hat;    // post-RMSNorm [d_in]
        float inv_rms_x = 1.0f;

        // Per-head normalized q, k, raw norms, v, delta error e, gates
        std::vector<float> q;        // [D]
        std::vector<float> k;        // [D]
        std::vector<float> inv_nq;   // [num_heads]
        std::vector<float> inv_nk;   // [num_heads]
        std::vector<float> v;        // [D]
        std::vector<float> e_err;    // v_t - S_{t-1} k_{t-1} [D]
        std::vector<float> alpha;    // [num_heads]
        std::vector<float> beta;     // [num_heads]

        // Associative matrix state S_t AFTER step t [num_heads * d_head * d_head]
        std::vector<float> S_after;

        // Output readout cache
        std::vector<float> y_un;     // RMSNorm normalized y_t [D]
        float inv_rms_y = 1.0f;

        // External gradients from readout loss at step t (predicting b_{t+1})
        std::vector<float> dy_ext;   // [D]
        std::vector<float> dx_ext;   // [d_in]
    };

    void forward_step(std::uint8_t in_byte, std::uint16_t tok_id);
    void run_bptt();
    void build_tree();

    std::size_t num_heads_;
    std::size_t dh_;
    std::size_t D_;          // num_heads_ * dh_
    std::size_t d_in_;
    std::size_t total_H_;    // D_ + d_in_
    std::size_t horizon_;
    std::size_t stride_;
    std::size_t step_ = 0;

    // Embeddings
    std::vector<float> e_byte_, m_ebyte_, v_ebyte_;   // [256 * d_in]
    std::vector<float> e_tok_, v_etok_;               // [kTokBuckets * d_in]

    // 4-tap depthwise causal convolution [kConvK * d_in] + bias [d_in]
    std::vector<float> conv_w_, m_convw_, v_convw_, g_convw_;
    std::vector<float> conv_b_, m_convb_, v_convb_, g_convb_;
    std::vector<float> e_ring_;  // last kConvK embeddings [kConvK * d_in]

    // Input RMSNorm [d_in] & Output RMSNorm [D]
    std::vector<float> gx_, m_gx_, v_gx_, g_gx_;
    std::vector<float> bx_, m_bx_, v_bx_, g_bx_;
    std::vector<float> gy_, m_gy_, v_gy_, g_gy_;
    std::vector<float> by_, m_by_, v_by_, g_by_;

    // Projections: W_Q, W_K, W_V [D * d_in]
    std::vector<float> wq_, m_wq_, v_wq_, g_wq_;
    std::vector<float> wk_, m_wk_, v_wk_, g_wk_;
    std::vector<float> wv_, m_wv_, v_wv_, g_wv_;

    // Per-head gate projections: W_alpha, W_beta [num_heads * d_in] + biases [num_heads]
    std::vector<float> wa_, m_wa_, v_wa_, g_wa_;
    std::vector<float> ba_, m_ba_, v_ba_, g_ba_;
    std::vector<float> wb_, m_wb_, v_wb_, g_wb_;
    std::vector<float> bb_, m_bb_, v_bb_, g_bb_;

    // Live associative matrix state S_cur [num_heads * dh * dh] & previous key k_prev [D]
    std::vector<float> S_cur_;
    std::vector<float> k_prev_;

    // Readout [256 * total_H] + bias [256]
    std::vector<float> wout_, m_wout_, v_wout_;
    std::vector<float> bout_, m_bout_, v_bout_;
    std::vector<float> h_concat_;
    std::array<float, 256> probs_{};

    // BPTT history ring buffer
    std::vector<StepCache> hist_;
    std::size_t hist_pos_ = 0;
    std::size_t hist_filled_ = 0;

    // Scratch buffers
    std::vector<float> dh_all_;
    std::vector<float> dS_;
    std::vector<float> dk_next_;
    std::vector<float> dq_raw_, dk_raw_, dv_;
    std::vector<float> dx_hat_, du_conv_;
    std::vector<float> de_shift_;  // [kConvK * d_in] backward conv gradient shift buffer

    std::array<std::uint32_t, 512> tree_{};
    std::uint32_t conf_bucket_ = 0u;
    bool avx2_ = false;
};

}  // namespace fzip::fzac_v3
