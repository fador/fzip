// fzip — Induction-Head Online Causal Self-Attention Transformer Predictor
// Combines 4-tap causal depthwise conv, Induction-Head KV alignment
// (K_t = context BEFORE b_t, V_t = observed b_t, Q_t = current context),
// QK-Norm + learned per-head relative position biases over W=64 window,
// SiLU FFN, exact Transformer-XL chunk BPTT, and pure bit-exact AVX2/scalar.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fzip::fzac_v3 {

class TransformerModel {
  public:
    // `d_model` multiple of 8 (default 64), `num_heads` (default 4 -> d_head=16),
    // `window` KV-cache attention length (default 64), `chunk_len` BPTT chunk (default 16).
    TransformerModel(std::size_t d_model = 64u, std::size_t num_heads = 4u,
                     std::size_t window = 64u, std::size_t chunk_len = 16u);

    void start_byte() {}

    // Return exact conditional probability P(bit == 1 | prefix `c0`) in [1, 4095].
    auto predict_bit(std::uint32_t c0) const -> std::uint32_t;

    // 2-bit prediction confidence / peakedness bucket (0..3) from current softmax.
    auto conf_bucket() const -> std::uint32_t { return conf_bucket_; }

    // Observe `actual_byte` (and hashed bigram/WRT token ID `tok_id`),
    // perform online Adam readout update + periodic exact chunk BPTT,
    // and advance Induction-Head causal KV-cache forward.
    void update_byte(std::uint8_t actual_byte, std::uint32_t tok_id = 0u);

  private:
    static constexpr std::size_t kConvK      = 4u;
    static constexpr std::size_t kTokBuckets = 2048u;
    static constexpr std::size_t kTokMask    = kTokBuckets - 1u;

    struct StepCache {
        std::uint8_t in_byte = 0;
        std::uint16_t tok_id = 0;
        std::size_t kv_slot = 0;
        std::size_t att_len = 0;  // number of valid keys attended (1..window_)

        // 4-tap conv input history e_{t-0..3} [kConvK * D]
        std::vector<float> conv_e;
        std::vector<float> conv_u;   // pre-SiLU [D]
        std::vector<float> x_un;     // post-SiLU normalized [D]
        std::vector<float> x_hat;    // current 4-byte context x_t [D]
        std::vector<float> x_prev;   // previous 4-byte context x_{t-1} (used for K_t!) [D]
        float inv_rms_x = 1.0f;

        // Induction-Head QK-normalized attention activations
        std::vector<float> q;        // L2-normalized per head [D]
        std::vector<float> inv_nq;   // [num_heads]
        std::vector<float> inv_nk;   // [num_heads]
        std::vector<float> attn;     // [num_heads * window_]
        std::vector<float> ctx;      // [D]

        // Post-attention / pre-FFN RMSNorm
        std::vector<float> x1_un;    // [D]
        std::vector<float> x1_norm;  // [D]
        float inv_rms1 = 1.0f;

        // FFN hidden activations [ffn_dim]
        std::vector<float> ffn_u;    // pre-SiLU
        std::vector<float> ffn_h;    // post-SiLU

        // Output RMSNorm
        std::vector<float> x2_un;    // [D]
        float inv_rms2 = 1.0f;

        // External gradient from readout loss predicting b_{t+1}
        std::vector<float> dx_ext;   // [D]
    };

    void forward_step(std::uint8_t in_byte, std::uint16_t tok_id);
    void run_chunk_bptt();
    void build_tree();

    std::size_t D_;
    std::size_t num_heads_;
    std::size_t dh_;
    std::size_t ffn_dim_;
    std::size_t window_;
    std::size_t chunk_len_;
    std::size_t kv_cap_;   // window_ + chunk_len_
    std::size_t step_ = 0;

    // Embeddings
    std::vector<float> e_byte_, m_ebyte_, v_ebyte_;  // [256 * D]
    std::vector<float> e_tok_, v_etok_;              // [kTokBuckets * D]

    // 4-tap depthwise causal conv [kConvK * D] + bias [D]
    std::vector<float> conv_w_, m_convw_, v_convw_, g_convw_;
    std::vector<float> conv_b_, m_convb_, v_convb_, g_convb_;
    std::vector<float> e_ring_;  // [kConvK * D]
    std::vector<float> x_prev_live_;  // x_{t-1} [D]

    // Input, Pre-FFN, and Output RMSNorm
    std::vector<float> gx_, m_gx_, v_gx_, g_gx_, bx_, m_bx_, v_bx_, g_bx_;
    std::vector<float> g1_, m_g1_, v_g1_, grad_g1_, b1_, m_b1_, v_b1_, grad_b1_;
    std::vector<float> g2_, m_g2_, v_g2_, grad_g2_, b2_, m_b2_, v_b2_, grad_b2_;

    // Attention weights: W_Q, W_K, W_V, W_O [D * D] + learned relative position bias [num_heads * window]
    std::vector<float> wq_, m_wq_, v_wq_, g_wq_;
    std::vector<float> wk_, m_wk_, v_wk_, g_wk_;
    std::vector<float> wv_, m_wv_, v_wv_, g_wv_;
    std::vector<float> wo_, m_wo_, v_wo_, g_wo_;
    std::vector<float> rel_b_, m_relb_, v_relb_, g_relb_;

    // FFN weights: W_1 [ffn_dim * D] + bf1 [ffn_dim], W_2 [D * ffn_dim] + bf2 [D]
    std::vector<float> w1_, m_w1_, v_w1_, g_w1_;
    std::vector<float> bf1_, m_bf1_, v_bf1_, g_bf1_;
    std::vector<float> w2_, m_w2_, v_w2_, g_w2_;
    std::vector<float> bf2_, m_bf2_, v_bf2_, g_bf2_;

    // Sliding KV cache [kv_cap * D] (k_cache_ stores L2-normalized keys!)
    std::vector<float> k_cache_;
    std::vector<float> v_cache_;
    std::size_t kv_pos_ = 0;
    std::size_t kv_filled_ = 0;

    // 256-way Readout [256 * D] + bias [256]
    std::vector<float> wout_, m_wout_, v_wout_;
    std::vector<float> bout_, m_bout_, v_bout_;
    std::vector<float> h_out_;
    std::array<float, 256> probs_{};

    // Chunk BPTT cache [chunk_len_]
    std::vector<StepCache> chunk_;
    std::size_t chunk_pos_ = 0;
    std::size_t chunk_filled_ = 0;

    // Scratch buffers for BPTT
    std::vector<float> dk_chunk_;   // [chunk_len_ * D]
    std::vector<float> dv_chunk_;   // [chunk_len_ * D]
    std::vector<float> dx2_, dx1_, dx_hat_, dx_from_next_k_, d_ctx_, dq_, dk_raw_, d_ffn_h_, d_ffn_u_;
    std::vector<float> de_shift_;   // [kConvK * D]

    std::array<std::uint32_t, 512> tree_{};
    std::uint32_t conf_bucket_ = 0u;
    bool avx2_ = false;
};

}  // namespace fzip::fzac_v3
