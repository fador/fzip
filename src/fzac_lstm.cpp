// fzip — Upgraded Multi-Layer Online Byte-Level Coupled-Gate LSTM Neural
// Predictor with RMSNorm, Adam Momentum, and Sparse Bigram/WRT Token Embeddings
// (bit-exact AVX2 + scalar fallback) for FZAC v3.
#include "fzac_lstm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#define FZAC_LSTM_X64 1
#else
#define FZAC_LSTM_X64 0
#endif

namespace fzip::fzac_v3 {

namespace {

auto detect_avx2_lstm() -> bool {
#if FZAC_LSTM_X64
#if defined(_MSC_VER)
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    if ((info[2] & (1 << 27)) == 0) return false;
    if ((_xgetbv(0) & 0x6ULL) != 0x6ULL) return false;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
#else
    return false;
#endif
}

inline auto fast_exp_scalar(float x) -> float {
    x = std::clamp(x, -20.0f, 20.0f);
    const float t = x * 1.4426950408889634f;
    const float kf = std::floor(t);
    const float f = t - kf;
    const int k = static_cast<int>(kf);
    const float p = 1.0f + f * (0.69314718f + f * (0.24022651f + f * (0.05550411f + f * 0.00961813f)));
    const std::int32_t bits = (k + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, 4);
    return p * scale;
}

#if FZAC_LSTM_X64
inline auto fast_exp_avx2(__m256 x) -> __m256 {
    x = _mm256_max_ps(_mm256_set1_ps(-20.0f), _mm256_min_ps(_mm256_set1_ps(20.0f), x));
    const __m256 t = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    const __m256 kf = _mm256_floor_ps(t);
    const __m256 f = _mm256_sub_ps(t, kf);
    const __m256i ki = _mm256_cvtps_epi32(kf);
    const __m256i bits = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    const __m256 scale = _mm256_castsi256_ps(bits);

    __m256 p = _mm256_add_ps(_mm256_mul_ps(f, _mm256_set1_ps(0.00961813f)), _mm256_set1_ps(0.05550411f));
    p = _mm256_add_ps(_mm256_mul_ps(f, p), _mm256_set1_ps(0.24022651f));
    p = _mm256_add_ps(_mm256_mul_ps(f, p), _mm256_set1_ps(0.69314718f));
    p = _mm256_add_ps(_mm256_mul_ps(f, p), _mm256_set1_ps(1.0f));
    return _mm256_mul_ps(p, scale);
}

inline auto fast_sigmoid_avx2(__m256 x) -> __m256 {
    const __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    const __m256 e = fast_exp_avx2(neg_x);
    return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

inline auto fast_tanh_avx2(__m256 x) -> __m256 {
    const __m256 s = fast_sigmoid_avx2(_mm256_mul_ps(x, _mm256_set1_ps(2.0f)));
    return _mm256_sub_ps(_mm256_mul_ps(s, _mm256_set1_ps(2.0f)), _mm256_set1_ps(1.0f));
}

inline auto hsum256_exact(__m256 v) -> float {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    const __m128 s4 = _mm_add_ps(lo, hi);
    const __m128 s2 = _mm_add_ps(
        _mm_shuffle_ps(s4, s4, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm_shuffle_ps(s4, s4, _MM_SHUFFLE(3, 1, 3, 1)));
    const __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(s1);
}
#endif

// Exact 8-lane dot product matching scalar & AVX2 tree order
inline auto dot_exact(const float* a, const float* b, std::size_t n, bool use_avx2) -> float {
#if FZAC_LSTM_X64
    if (use_avx2) {
        __m256 acc = _mm256_setzero_ps();
        for (std::size_t c = 0; c < n; c += 8u) {
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + c), _mm256_loadu_ps(b + c)));
        }
        return hsum256_exact(acc);
    }
#else
    (void)use_avx2;
#endif
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (std::size_t c = 0; c < n; c += 8u) {
        for (std::size_t k = 0; k < 8u; ++k) {
            acc[k] = acc[k] + (a[c + k] * b[c + k]);
        }
    }
    return ((acc[0] + acc[4]) + (acc[1] + acc[5])) +
           ((acc[2] + acc[6]) + (acc[3] + acc[7]));
}

void matvec_add_exact(float* out, const float* M, const float* x,
                      std::size_t rows, std::size_t cols, bool use_avx2) {
#if FZAC_LSTM_X64
    if (use_avx2) {
        std::size_t r = 0;
        for (; r + 4u <= rows; r += 4u) {
            const float* r0 = M + (r + 0u) * cols;
            const float* r1 = M + (r + 1u) * cols;
            const float* r2 = M + (r + 2u) * cols;
            const float* r3 = M + (r + 3u) * cols;
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            for (std::size_t c = 0; c < cols; c += 8u) {
                const __m256 vx = _mm256_loadu_ps(x + c);
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(r0 + c), vx));
                a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(r1 + c), vx));
                a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(r2 + c), vx));
                a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(r3 + c), vx));
            }
            out[r + 0u] += hsum256_exact(a0);
            out[r + 1u] += hsum256_exact(a1);
            out[r + 2u] += hsum256_exact(a2);
            out[r + 3u] += hsum256_exact(a3);
        }
        for (; r < rows; ++r) {
            const float* row = M + r * cols;
            __m256 acc = _mm256_setzero_ps();
            for (std::size_t c = 0; c < cols; c += 8u) {
                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row + c),
                                                       _mm256_loadu_ps(x + c)));
            }
            out[r] += hsum256_exact(acc);
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t r = 0; r < rows; ++r) {
        const float* row = M + r * cols;
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (std::size_t c = 0; c < cols; c += 8u) {
            for (std::size_t k = 0; k < 8u; ++k) {
                acc[k] = acc[k] + (row[c + k] * x[c + k]);
            }
        }
        out[r] += ((acc[0] + acc[4]) + (acc[1] + acc[5])) +
                  ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    }
}

// RMSNorm forward: u = a / rms(a), y = gamma * u + bias
void rmsnorm_forward(const float* a, const float* gamma, const float* bias,
                     float* u_out, float* y_out, float& inv_rms_out,
                     std::size_t H, bool use_avx2) {
    const float inv_H = 1.0f / static_cast<float>(H);
    const float ss = dot_exact(a, a, H, use_avx2) * inv_H + 1e-4f;
    const float r = 1.0f / std::sqrt(ss);
    inv_rms_out = r;
#if FZAC_LSTM_X64
    if (use_avx2) {
        const __m256 vr = _mm256_set1_ps(r);
        for (std::size_t i = 0; i < H; i += 8u) {
            const __m256 vu = _mm256_mul_ps(_mm256_loadu_ps(a + i), vr);
            _mm256_storeu_ps(u_out + i, vu);
            const __m256 vy = _mm256_add_ps(
                _mm256_mul_ps(vu, _mm256_loadu_ps(gamma + i)),
                _mm256_loadu_ps(bias + i));
            _mm256_storeu_ps(y_out + i, vy);
        }
        return;
    }
#endif
    for (std::size_t i = 0; i < H; ++i) {
        const float u = a[i] * r;
        u_out[i] = u;
        y_out[i] = u * gamma[i] + bias[i];
    }
}

// RMSNorm backward: given dy (in-place overwritten with da), accumulates g_gamma & g_bias
void rmsnorm_backward_inplace(float* d_io, const float* u, const float* gamma,
                              float inv_rms, float* g_gamma, float* g_bias,
                              std::size_t H, bool use_avx2) {
    const float inv_H = 1.0f / static_cast<float>(H);
    float du_tmp[128];
#if FZAC_LSTM_X64
    if (use_avx2) {
        __m256 vdot = _mm256_setzero_ps();
        for (std::size_t i = 0; i < H; i += 8u) {
            const __m256 vdy = _mm256_loadu_ps(d_io + i);
            const __m256 vu = _mm256_loadu_ps(u + i);
            const __m256 vg = _mm256_loadu_ps(gamma + i);
            _mm256_storeu_ps(g_bias + i, _mm256_add_ps(_mm256_loadu_ps(g_bias + i), vdy));
            _mm256_storeu_ps(g_gamma + i, _mm256_add_ps(_mm256_loadu_ps(g_gamma + i), _mm256_mul_ps(vdy, vu)));
            const __m256 vdu = _mm256_mul_ps(vdy, vg);
            _mm256_storeu_ps(du_tmp + i, vdu);
            vdot = _mm256_add_ps(vdot, _mm256_mul_ps(vdu, vu));
        }
        const float mean_du_u = hsum256_exact(vdot) * inv_H;
        const __m256 vmean = _mm256_set1_ps(mean_du_u);
        const __m256 vinv = _mm256_set1_ps(inv_rms);
        for (std::size_t i = 0; i < H; i += 8u) {
            const __m256 vdu = _mm256_loadu_ps(du_tmp + i);
            const __m256 vu = _mm256_loadu_ps(u + i);
            const __m256 vda = _mm256_mul_ps(vinv, _mm256_sub_ps(vdu, _mm256_mul_ps(vu, vmean)));
            _mm256_storeu_ps(d_io + i, vda);
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (std::size_t i = 0; i < H; i += 8u) {
        for (std::size_t k = 0; k < 8u; ++k) {
            const std::size_t j = i + k;
            const float dy = d_io[j];
            g_bias[j] = g_bias[j] + dy;
            g_gamma[j] = g_gamma[j] + (dy * u[j]);
            const float du = dy * gamma[j];
            du_tmp[j] = du;
            acc[k] = acc[k] + (du * u[j]);
        }
    }
    const float mean_du_u =
        (((acc[0] + acc[4]) + (acc[1] + acc[5])) +
         ((acc[2] + acc[6]) + (acc[3] + acc[7]))) * inv_H;
    for (std::size_t i = 0; i < H; ++i) {
        d_io[i] = inv_rms * (du_tmp[i] - u[i] * mean_du_u);
    }
}

void compute_gates_and_state(const float* af, const float* ag, const float* ao,
                             const float* c_prev,
                             float* f_out, float* g_out, float* o_out,
                             float* tc_out, float* c_out, float* h_out,
                             std::size_t H, bool use_avx2) {
#if FZAC_LSTM_X64
    if (use_avx2) {
        const __m256 vone = _mm256_set1_ps(1.0f);
        for (std::size_t i = 0; i < H; i += 8u) {
            const __m256 vf = fast_sigmoid_avx2(_mm256_loadu_ps(af + i));
            const __m256 vg = fast_tanh_avx2(_mm256_loadu_ps(ag + i));
            const __m256 vo = fast_sigmoid_avx2(_mm256_loadu_ps(ao + i));
            const __m256 vcp = _mm256_loadu_ps(c_prev + i);
            const __m256 vc = _mm256_add_ps(_mm256_mul_ps(vf, vcp),
                                            _mm256_mul_ps(_mm256_sub_ps(vone, vf), vg));
            const __m256 vtc = fast_tanh_avx2(vc);
            const __m256 vh = _mm256_mul_ps(vo, vtc);
            _mm256_storeu_ps(f_out + i, vf);
            _mm256_storeu_ps(g_out + i, vg);
            _mm256_storeu_ps(o_out + i, vo);
            _mm256_storeu_ps(tc_out + i, vtc);
            _mm256_storeu_ps(c_out + i, vc);
            _mm256_storeu_ps(h_out + i, vh);
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t i = 0; i < H; ++i) {
        const float f = 1.0f / (1.0f + fast_exp_scalar(-af[i]));
        const float g = 2.0f / (1.0f + fast_exp_scalar(-2.0f * ag[i])) - 1.0f;
        const float o = 1.0f / (1.0f + fast_exp_scalar(-ao[i]));
        const float c = f * c_prev[i] + (1.0f - f) * g;
        const float tc = 2.0f / (1.0f + fast_exp_scalar(-2.0f * c)) - 1.0f;
        f_out[i] = f;
        g_out[i] = g;
        o_out[i] = o;
        tc_out[i] = tc;
        c_out[i] = c;
        h_out[i] = o * tc;
    }
}

void softmax_256(const float* logits, float* probs, bool use_avx2) {
    float max_l = logits[0];
    for (std::size_t k = 1u; k < 256u; ++k) {
        if (logits[k] > max_l) max_l = logits[k];
    }
    float sum_e = 0.0f;
#if FZAC_LSTM_X64
    if (use_avx2) {
        const __m256 vmax = _mm256_set1_ps(max_l);
        __m256 vsum = _mm256_setzero_ps();
        for (std::size_t k = 0; k < 256u; k += 8u) {
            const __m256 ve = fast_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(logits + k), vmax));
            _mm256_storeu_ps(probs + k, ve);
            vsum = _mm256_add_ps(vsum, ve);
        }
        sum_e = hsum256_exact(vsum);
        const __m256 vinv = _mm256_set1_ps(1.0f / sum_e);
        for (std::size_t k = 0; k < 256u; k += 8u) {
            _mm256_storeu_ps(probs + k, _mm256_mul_ps(_mm256_loadu_ps(probs + k), vinv));
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (std::size_t k = 0; k < 256u; k += 8u) {
        for (std::size_t j = 0; j < 8u; ++j) {
            const float e = fast_exp_scalar(logits[k + j] - max_l);
            probs[k + j] = e;
            acc[j] = acc[j] + e;
        }
    }
    sum_e = ((acc[0] + acc[4]) + (acc[1] + acc[5])) +
            ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    const float inv_sum = 1.0f / sum_e;
    for (std::size_t k = 0; k < 256u; ++k) {
        probs[k] = probs[k] * inv_sum;
    }
}

void bptt_mat_step(float* gW, const float* W, const float* d_gate,
                   const float* x_in, float* dx_out,
                   std::size_t rows, std::size_t cols, bool use_avx2) {
#if FZAC_LSTM_X64
    if (use_avx2) {
        for (std::size_t r = 0; r < rows; ++r) {
            const __m256 vd = _mm256_set1_ps(d_gate[r]);
            float* grow = gW + r * cols;
            const float* wrow = W + r * cols;
            for (std::size_t c = 0; c < cols; c += 8u) {
                const __m256 vx = _mm256_loadu_ps(x_in + c);
                const __m256 vg = _mm256_loadu_ps(grow + c);
                _mm256_storeu_ps(grow + c, _mm256_add_ps(vg, _mm256_mul_ps(vx, vd)));

                const __m256 vw = _mm256_loadu_ps(wrow + c);
                const __m256 vdx = _mm256_loadu_ps(dx_out + c);
                _mm256_storeu_ps(dx_out + c, _mm256_add_ps(vdx, _mm256_mul_ps(vw, vd)));
            }
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t r = 0; r < rows; ++r) {
        const float d = d_gate[r];
        float* grow = gW + r * cols;
        const float* wrow = W + r * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            grow[c] = grow[c] + (x_in[c] * d);
            dx_out[c] = dx_out[c] + (wrow[c] * d);
        }
    }
}

// Adam output-layer sparse update (beta1=0.50, beta2=0.99) in contiguous AVX2!
void update_output_layer_adam(float* wout, float* mwout, float* vwout,
                              float* bout, float* mbout, float* vbout,
                              const float* h_concat, const float* probs,
                              std::uint8_t actual_byte, float* dh_out,
                              float lr, std::size_t total_H, bool use_avx2) {
    constexpr float kB1 = 0.50f, k1mB1 = 0.50f;
    constexpr float kB2 = 0.99f, k1mB2 = 0.01f;
    constexpr float kEps = 1e-4f;
    std::memset(dh_out, 0, total_H * sizeof(float));

    // Pre-collect active classes so AVX2 wout loop has ZERO scalar SSE transitions!
    std::uint16_t active_k[256];
    float active_dl[256];
    std::size_t n_act = 0u;
    for (std::size_t k = 0; k < 256u; ++k) {
        const float dl = probs[k] - (k == actual_byte ? 1.0f : 0.0f);
        if (dl > 4e-4f || dl < -4e-4f) {
            active_k[n_act] = static_cast<std::uint16_t>(k);
            active_dl[n_act] = dl;
            ++n_act;
        }
    }

    // Update biases in scalar pass BEFORE entering AVX2 region
    for (std::size_t idx = 0; idx < n_act; ++idx) {
        const std::size_t k = active_k[idx];
        const float g = std::clamp(active_dl[idx], -2.0f, 2.0f);
        const float mm = mbout[k] * kB1 + g * k1mB1;
        const float vv = vbout[k] * kB2 + (g * g) * k1mB2;
        mbout[k] = mm;
        vbout[k] = vv;
        bout[k] = std::clamp(bout[k] - (lr * mm) / (std::sqrt(vv) + kEps), -8.0f, 8.0f);
    }

#if FZAC_LSTM_X64
    if (use_avx2) {
        const __m256 vlr = _mm256_set1_ps(lr);
        const __m256 vb1 = _mm256_set1_ps(kB1);
        const __m256 v1mb1 = _mm256_set1_ps(k1mB1);
        const __m256 vb2 = _mm256_set1_ps(kB2);
        const __m256 v1mb2 = _mm256_set1_ps(k1mB2);
        const __m256 veps = _mm256_set1_ps(kEps);
        const __m256 vgmin = _mm256_set1_ps(-2.0f);
        const __m256 vgmax = _mm256_set1_ps(2.0f);
        const __m256 vwmin = _mm256_set1_ps(-8.0f);
        const __m256 vwmax = _mm256_set1_ps(8.0f);

        for (std::size_t idx = 0; idx < n_act; ++idx) {
            const std::size_t k = active_k[idx];
            const __m256 vdl = _mm256_set1_ps(active_dl[idx]);
            float* wrow = wout + k * total_H;
            float* mrow = mwout + k * total_H;
            float* vrow = vwout + k * total_H;
            for (std::size_t c = 0; c < total_H; c += 8u) {
                const __m256 vw = _mm256_loadu_ps(wrow + c);
                const __m256 vdh = _mm256_loadu_ps(dh_out + c);
                _mm256_storeu_ps(dh_out + c, _mm256_add_ps(vdh, _mm256_mul_ps(vw, vdl)));

                const __m256 vh = _mm256_loadu_ps(h_concat + c);
                __m256 vg = _mm256_mul_ps(vh, vdl);
                vg = _mm256_max_ps(vgmin, _mm256_min_ps(vgmax, vg));
                const __m256 vm = _mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(mrow + c), vb1),
                    _mm256_mul_ps(vg, v1mb1));
                _mm256_storeu_ps(mrow + c, vm);
                const __m256 vv = _mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(vrow + c), vb2),
                    _mm256_mul_ps(_mm256_mul_ps(vg, vg), v1mb2));
                _mm256_storeu_ps(vrow + c, vv);
                const __m256 step = _mm256_div_ps(
                    _mm256_mul_ps(vlr, vm), _mm256_add_ps(_mm256_sqrt_ps(vv), veps));
                __m256 wnew = _mm256_sub_ps(vw, step);
                wnew = _mm256_max_ps(vwmin, _mm256_min_ps(vwmax, wnew));
                _mm256_storeu_ps(wrow + c, wnew);
            }
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t idx = 0; idx < n_act; ++idx) {
        const std::size_t k = active_k[idx];
        const float dl = active_dl[idx];
        float* wrow = wout + k * total_H;
        float* mrow = mwout + k * total_H;
        float* vrow = vwout + k * total_H;
        for (std::size_t c = 0; c < total_H; ++c) {
            dh_out[c] = dh_out[c] + (wrow[c] * dl);
            const float g = std::clamp(h_concat[c] * dl, -2.0f, 2.0f);
            const float mm = mrow[c] * kB1 + g * k1mB1;
            const float vv = vrow[c] * kB2 + (g * g) * k1mB2;
            mrow[c] = mm;
            vrow[c] = vv;
            wrow[c] = std::clamp(wrow[c] - (lr * mm) / (std::sqrt(vv) + kEps), -8.0f, 8.0f);
        }
    }
}

void adam_update_grad(float* w, float* m, float* v, float* grad,
                      float lr, std::size_t n, bool use_avx2, bool clear_grad = true) {
    constexpr float kB1 = 0.50f, k1mB1 = 0.50f;
    constexpr float kB2 = 0.99f, k1mB2 = 0.01f;
    constexpr float kEps = 1e-4f;
#if FZAC_LSTM_X64
    if (use_avx2) {
        const __m256 vlr = _mm256_set1_ps(lr);
        const __m256 vb1 = _mm256_set1_ps(kB1);
        const __m256 v1mb1 = _mm256_set1_ps(k1mB1);
        const __m256 vb2 = _mm256_set1_ps(kB2);
        const __m256 v1mb2 = _mm256_set1_ps(k1mB2);
        const __m256 veps = _mm256_set1_ps(kEps);
        const __m256 vgmin = _mm256_set1_ps(-2.0f);
        const __m256 vgmax = _mm256_set1_ps(2.0f);
        const __m256 vwmin = _mm256_set1_ps(-8.0f);
        const __m256 vwmax = _mm256_set1_ps(8.0f);
        const __m256 vzero = _mm256_setzero_ps();
        for (std::size_t i = 0; i < n; i += 8u) {
            __m256 vg = _mm256_loadu_ps(grad + i);
            if (clear_grad) _mm256_storeu_ps(grad + i, vzero);
            vg = _mm256_max_ps(vgmin, _mm256_min_ps(vgmax, vg));
            const __m256 vm = _mm256_add_ps(
                _mm256_mul_ps(_mm256_loadu_ps(m + i), vb1),
                _mm256_mul_ps(vg, v1mb1));
            _mm256_storeu_ps(m + i, vm);
            const __m256 vv = _mm256_add_ps(
                _mm256_mul_ps(_mm256_loadu_ps(v + i), vb2),
                _mm256_mul_ps(_mm256_mul_ps(vg, vg), v1mb2));
            _mm256_storeu_ps(v + i, vv);
            const __m256 step = _mm256_div_ps(
                _mm256_mul_ps(vlr, vm), _mm256_add_ps(_mm256_sqrt_ps(vv), veps));
            __m256 vw = _mm256_sub_ps(_mm256_loadu_ps(w + i), step);
            vw = _mm256_max_ps(vwmin, _mm256_min_ps(vwmax, vw));
            _mm256_storeu_ps(w + i, vw);
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t i = 0; i < n; ++i) {
        const float g = std::clamp(grad[i], -2.0f, 2.0f);
        if (clear_grad) grad[i] = 0.0f;
        const float mm = m[i] * kB1 + g * k1mB1;
        const float vv = v[i] * kB2 + (g * g) * k1mB2;
        m[i] = mm;
        v[i] = vv;
        w[i] = std::clamp(w[i] - (lr * mm) / (std::sqrt(vv) + kEps), -8.0f, 8.0f);
    }
}

void rmsprop_update_sparse(float* w, float* v, const float* grad,
                           float lr, std::size_t n, bool use_avx2) {
    constexpr float kB2 = 0.99f, k1mB2 = 0.01f, kEps = 1e-4f;
#if FZAC_LSTM_X64
    if (use_avx2) {
        const __m256 vlr = _mm256_set1_ps(lr);
        const __m256 vb2 = _mm256_set1_ps(kB2);
        const __m256 v1mb2 = _mm256_set1_ps(k1mB2);
        const __m256 veps = _mm256_set1_ps(kEps);
        const __m256 vgmin = _mm256_set1_ps(-2.0f);
        const __m256 vgmax = _mm256_set1_ps(2.0f);
        const __m256 vwmin = _mm256_set1_ps(-4.0f);
        const __m256 vwmax = _mm256_set1_ps(4.0f);
        for (std::size_t i = 0; i < n; i += 8u) {
            __m256 vg = _mm256_loadu_ps(grad + i);
            vg = _mm256_max_ps(vgmin, _mm256_min_ps(vgmax, vg));
            const __m256 vv = _mm256_add_ps(
                _mm256_mul_ps(_mm256_loadu_ps(v + i), vb2),
                _mm256_mul_ps(_mm256_mul_ps(vg, vg), v1mb2));
            _mm256_storeu_ps(v + i, vv);
            const __m256 step = _mm256_div_ps(
                _mm256_mul_ps(vlr, vg), _mm256_add_ps(_mm256_sqrt_ps(vv), veps));
            __m256 vw = _mm256_sub_ps(_mm256_loadu_ps(w + i), step);
            vw = _mm256_max_ps(vwmin, _mm256_min_ps(vwmax, vw));
            _mm256_storeu_ps(w + i, vw);
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t i = 0; i < n; ++i) {
        const float g = std::clamp(grad[i], -2.0f, 2.0f);
        const float vv = v[i] * kB2 + (g * g) * k1mB2;
        v[i] = vv;
        w[i] = std::clamp(w[i] - (lr * g) / (std::sqrt(vv) + kEps), -4.0f, 4.0f);
    }
}

struct XorShift32 {
    std::uint32_t s = 0x243F6A88u;
    auto next_f() -> float {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return static_cast<float>(static_cast<std::int32_t>(s)) * (1.0f / 2147483648.0f);
    }
};

}  // namespace

LstmModel::LstmModel(std::size_t num_layers, std::size_t hidden_size, std::size_t horizon)
    : num_layers_(std::clamp<std::size_t>(num_layers, 1u, 2u)),
      H_(std::clamp<std::size_t>((hidden_size + 7u) & ~std::size_t{7}, 16u, 128u)),
      total_H_(num_layers_ * H_),
      horizon_(std::max<std::size_t>(8u, horizon)),
      stride_(horizon_),
      avx2_(detect_avx2_lstm()) {
    XorShift32 rng;
    const float r_emb = 0.25f;
    const float r_rec = 1.0f / std::sqrt(static_cast<float>(H_));
    const std::size_t eh = 256u * H_;
    const std::size_t hh = H_ * H_;

    layers_.resize(num_layers_);
    for (std::size_t l = 0; l < num_layers_; ++l) {
        auto& ly = layers_[l];
        ly.in_dim = (l == 0u) ? 0u : H_;

        ly.ef.resize(eh); ly.eg.resize(eh); ly.eo.resize(eh);
        ly.m_ef.assign(eh, 0.0f); ly.m_eg.assign(eh, 0.0f); ly.m_eo.assign(eh, 0.0f);
        ly.v_ef.assign(eh, 1e-3f); ly.v_eg.assign(eh, 1e-3f); ly.v_eo.assign(eh, 1e-3f);
        for (std::size_t i = 0; i < eh; ++i) {
            ly.ef[i] = rng.next_f() * r_emb;
            ly.eg[i] = rng.next_f() * r_emb;
            ly.eo[i] = rng.next_f() * r_emb;
        }

        if (l == 0u) {
            const std::size_t th = kTokBuckets * H_;
            ly.tf.assign(th, 0.0f); ly.tg.assign(th, 0.0f); ly.to.assign(th, 0.0f);
            ly.v_tf.assign(th, 1e-3f); ly.v_tg.assign(th, 1e-3f); ly.v_to.assign(th, 1e-3f);
        }

        ly.wf.resize(hh); ly.wg.resize(hh); ly.wo.resize(hh);
        ly.m_wf.assign(hh, 0.0f); ly.m_wg.assign(hh, 0.0f); ly.m_wo.assign(hh, 0.0f);
        ly.v_wf.assign(hh, 1e-3f); ly.v_wg.assign(hh, 1e-3f); ly.v_wo.assign(hh, 1e-3f);
        ly.g_wf.assign(hh, 0.0f);  ly.g_wg.assign(hh, 0.0f);  ly.g_wo.assign(hh, 0.0f);
        for (std::size_t i = 0; i < hh; ++i) {
            ly.wf[i] = rng.next_f() * r_rec;
            ly.wg[i] = rng.next_f() * r_rec;
            ly.wo[i] = rng.next_f() * r_rec;
        }

        if (ly.in_dim > 0u) {
            const std::size_t uh = H_ * ly.in_dim;
            ly.uf.resize(uh); ly.ug.resize(uh); ly.uo.resize(uh);
            ly.m_uf.assign(uh, 0.0f); ly.m_ug.assign(uh, 0.0f); ly.m_uo.assign(uh, 0.0f);
            ly.v_uf.assign(uh, 1e-3f); ly.v_ug.assign(uh, 1e-3f); ly.v_uo.assign(uh, 1e-3f);
            ly.g_uf.assign(uh, 0.0f);  ly.g_ug.assign(uh, 0.0f);  ly.g_uo.assign(uh, 0.0f);
            for (std::size_t i = 0; i < uh; ++i) {
                ly.uf[i] = rng.next_f() * r_rec;
                ly.ug[i] = rng.next_f() * r_rec;
                ly.uo[i] = rng.next_f() * r_rec;
            }
        }

        ly.gf.assign(H_, 1.0f); ly.gg.assign(H_, 1.0f); ly.go.assign(H_, 1.0f);
        ly.m_gf.assign(H_, 0.0f); ly.m_gg.assign(H_, 0.0f); ly.m_go.assign(H_, 0.0f);
        ly.v_gf.assign(H_, 1e-3f); ly.v_gg.assign(H_, 1e-3f); ly.v_go.assign(H_, 1e-3f);
        ly.g_gf.assign(H_, 0.0f); ly.g_gg.assign(H_, 0.0f); ly.g_go.assign(H_, 0.0f);

        ly.bf.assign(H_, 0.6f + 0.4f * static_cast<float>(l));
        ly.bg.assign(H_, 0.0f);
        ly.bo.assign(H_, 0.0f);
        ly.m_bf.assign(H_, 0.0f); ly.m_bg.assign(H_, 0.0f); ly.m_bo.assign(H_, 0.0f);
        ly.v_bf.assign(H_, 1e-3f); ly.v_bg.assign(H_, 1e-3f); ly.v_bo.assign(H_, 1e-3f);
        ly.g_bf.assign(H_, 0.0f);  ly.g_bg.assign(H_, 0.0f);  ly.g_bo.assign(H_, 0.0f);
        ly.h_cur.assign(H_, 0.0f);
        ly.c_cur.assign(H_, 0.0f);
    }

    wout_.assign(256u * total_H_, 0.0f);
    m_wout_.assign(256u * total_H_, 0.0f);
    v_wout_.assign(256u * total_H_, 1e-3f);
    bout_.assign(256u, 0.0f);
    m_bout_.assign(256u, 0.0f);
    v_bout_.assign(256u, 1e-3f);
    h_concat_.assign(total_H_, 0.0f);
    probs_.fill(1.0f / 256.0f);

    hist_.resize(horizon_);
    for (auto& sc : hist_) {
        sc.ls.resize(num_layers_);
        for (auto& ls : sc.ls) {
            ls.h_prev.assign(H_, 0.0f);
            ls.c_prev.assign(H_, 0.0f);
            ls.uf_n.assign(H_, 0.0f);
            ls.ug_n.assign(H_, 0.0f);
            ls.uo_n.assign(H_, 0.0f);
            ls.f.assign(H_, 0.0f);
            ls.g.assign(H_, 0.0f);
            ls.o.assign(H_, 0.0f);
            ls.tanh_c.assign(H_, 0.0f);
            ls.h_out.assign(H_, 0.0f);
            ls.dh_ext.assign(H_, 0.0f);
        }
    }

    dh_all_.assign(total_H_, 0.0f);
    dh_next_.assign(H_, 0.0f);
    dc_next_.assign(H_, 0.0f);
    df_.assign(H_, 0.0f);
    dg_.assign(H_, 0.0f);
    do_.assign(H_, 0.0f);
    af_n_.assign(H_, 0.0f);
    ag_n_.assign(H_, 0.0f);
    ao_n_.assign(H_, 0.0f);
    dh_below_.assign(H_, 0.0f);

    build_tree();
}

void LstmModel::build_tree() {
    float max_p = 0.0f;
    for (std::size_t b = 0; b < 256u; ++b) {
        if (probs_[b] > max_p) max_p = probs_[b];
        const std::uint32_t v = static_cast<std::uint32_t>(probs_[b] * 16777216.0f);
        tree_[256u + b] = (v == 0u) ? 1u : v;
    }
    for (std::size_t i = 255u; i >= 1u; --i) {
        tree_[i] = tree_[i * 2u] + tree_[i * 2u + 1u];
    }
    conf_bucket_ = (max_p < 0.25f) ? 0u : ((max_p < 0.55f) ? 1u : ((max_p < 0.85f) ? 2u : 3u));
}

auto LstmModel::predict_bit(std::uint32_t c0) const -> std::uint32_t {
    const std::uint32_t tot = tree_[c0];
    const std::uint32_t p1 = tree_[c0 * 2u + 1u];
    const std::uint64_t p = (static_cast<std::uint64_t>(p1) * 4096u + (tot >> 1)) / tot;
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(p, 1u, 4095u));
}

void LstmModel::forward_step(std::uint8_t in_byte, std::uint16_t tok_id) {
    auto& sc = hist_[hist_pos_];
    sc.in_byte = in_byte;
    sc.tok_id = tok_id;

    for (std::size_t l = 0; l < num_layers_; ++l) {
        auto& ly = layers_[l];
        auto& ls = sc.ls[l];
        std::memcpy(ls.h_prev.data(), ly.h_cur.data(), H_ * sizeof(float));
        std::memcpy(ls.c_prev.data(), ly.c_cur.data(), H_ * sizeof(float));

        const std::size_t eoff = static_cast<std::size_t>(in_byte) * H_;
        if (l == 0u) {
            const std::size_t toff = static_cast<std::size_t>(tok_id) * H_;
            for (std::size_t i = 0; i < H_; ++i) {
                df_[i] = ly.ef[eoff + i] + ly.tf[toff + i];
                dg_[i] = ly.eg[eoff + i] + ly.tg[toff + i];
                do_[i] = ly.eo[eoff + i] + ly.to[toff + i];
            }
        } else {
            for (std::size_t i = 0; i < H_; ++i) {
                df_[i] = ly.ef[eoff + i];
                dg_[i] = ly.eg[eoff + i];
                do_[i] = ly.eo[eoff + i];
            }
        }

        matvec_add_exact(df_.data(), ly.wf.data(), ls.h_prev.data(), H_, H_, avx2_);
        matvec_add_exact(dg_.data(), ly.wg.data(), ls.h_prev.data(), H_, H_, avx2_);
        matvec_add_exact(do_.data(), ly.wo.data(), ls.h_prev.data(), H_, H_, avx2_);
        if (l > 0u) {
            const float* prev_layer_h = sc.ls[l - 1u].h_out.data();
            matvec_add_exact(df_.data(), ly.uf.data(), prev_layer_h, H_, H_, avx2_);
            matvec_add_exact(dg_.data(), ly.ug.data(), prev_layer_h, H_, H_, avx2_);
            matvec_add_exact(do_.data(), ly.uo.data(), prev_layer_h, H_, H_, avx2_);
        }

        // Apply RMSNorm to all 3 gate pre-activations for stability & faster convergence!
        rmsnorm_forward(df_.data(), ly.gf.data(), ly.bf.data(), ls.uf_n.data(), af_n_.data(), ls.r_f, H_, avx2_);
        rmsnorm_forward(dg_.data(), ly.gg.data(), ly.bg.data(), ls.ug_n.data(), ag_n_.data(), ls.r_g, H_, avx2_);
        rmsnorm_forward(do_.data(), ly.go.data(), ly.bo.data(), ls.uo_n.data(), ao_n_.data(), ls.r_o, H_, avx2_);

        compute_gates_and_state(af_n_.data(), ag_n_.data(), ao_n_.data(), ls.c_prev.data(),
                                ls.f.data(), ls.g.data(), ls.o.data(),
                                ls.tanh_c.data(), ly.c_cur.data(), ly.h_cur.data(),
                                H_, avx2_);
        std::memcpy(ls.h_out.data(), ly.h_cur.data(), H_ * sizeof(float));
        std::memcpy(h_concat_.data() + l * H_, ly.h_cur.data(), H_ * sizeof(float));
    }

    // Quantize top-layer hidden state (4 quadrants) into a 4-bit semantic regime ID for APM
    const auto& htop = layers_.back().h_cur;
    const std::size_t q = H_ >> 2;
    std::uint32_t sh = 0u;
    for (std::size_t b = 0; b < 4u; ++b) {
        float acc = 0.0f;
        const std::size_t base = b * q;
        for (std::size_t j = 0; j < q; ++j) acc += htop[base + j];
        if (acc > 0.0f) sh |= (1u << b);
    }
    state_hash_ = sh;

    std::array<float, 256> logits{};
    std::memcpy(logits.data(), bout_.data(), 256u * sizeof(float));
    matvec_add_exact(logits.data(), wout_.data(), h_concat_.data(), 256u, total_H_, avx2_);
    softmax_256(logits.data(), probs_.data(), avx2_);
    build_tree();
}

void LstmModel::run_bptt() {
    const std::size_t steps = std::min(hist_filled_, horizon_);
    const float decay = 1.0f / std::sqrt(1.0f + static_cast<float>(step_) * (1.0f / 262144.0f));
    const float lr_emb = ((step_ < 16384u) ? 0.045f : 0.028f) * decay;
    const float lr_tok = ((step_ < 16384u) ? 0.055f : 0.032f) * decay;
    const float lr_rec = ((step_ < 16384u) ? 0.026f : 0.015f) * decay;

    for (int l = static_cast<int>(num_layers_) - 1; l >= 0; --l) {
        auto& ly = layers_[static_cast<std::size_t>(l)];
        std::fill(dh_next_.begin(), dh_next_.end(), 0.0f);
        std::fill(dc_next_.begin(), dc_next_.end(), 0.0f);

        std::size_t idx = (hist_pos_ + horizon_ - 1u) % horizon_;
        for (std::size_t s = 0; s < steps; ++s) {
            auto& sc = hist_[idx];
            auto& ls = sc.ls[static_cast<std::size_t>(l)];

            for (std::size_t i = 0; i < H_; ++i) {
                const float dh = std::clamp(ls.dh_ext[i] + dh_next_[i], -1.5f, 1.5f);
                const float tc = ls.tanh_c[i];
                const float o = ls.o[i];
                const float f = ls.f[i];
                const float g = ls.g[i];

                const float dc = std::clamp(dh * o * (1.0f - tc * tc) + dc_next_[i], -1.5f, 1.5f);
                do_[i] = dh * tc * o * (1.0f - o);
                df_[i] = dc * (ls.c_prev[i] - g) * f * (1.0f - f);
                dg_[i] = dc * (1.0f - f) * (1.0f - g * g);
                dc_next_[i] = dc * f;
            }

            // Backprop through RMSNorm (accumulates g_gamma & g_bias, converts dy -> da)
            rmsnorm_backward_inplace(df_.data(), ls.uf_n.data(), ly.gf.data(), ls.r_f,
                                     ly.g_gf.data(), ly.g_bf.data(), H_, avx2_);
            rmsnorm_backward_inplace(dg_.data(), ls.ug_n.data(), ly.gg.data(), ls.r_g,
                                     ly.g_gg.data(), ly.g_bg.data(), H_, avx2_);
            rmsnorm_backward_inplace(do_.data(), ls.uo_n.data(), ly.go.data(), ls.r_o,
                                     ly.g_go.data(), ly.g_bo.data(), H_, avx2_);

            std::fill(dh_next_.begin(), dh_next_.end(), 0.0f);
            bptt_mat_step(ly.g_wf.data(), ly.wf.data(), df_.data(), ls.h_prev.data(), dh_next_.data(), H_, H_, avx2_);
            bptt_mat_step(ly.g_wg.data(), ly.wg.data(), dg_.data(), ls.h_prev.data(), dh_next_.data(), H_, H_, avx2_);
            bptt_mat_step(ly.g_wo.data(), ly.wo.data(), do_.data(), ls.h_prev.data(), dh_next_.data(), H_, H_, avx2_);

            if (l > 0) {
                auto& ls_below = sc.ls[static_cast<std::size_t>(l - 1)];
                std::fill(dh_below_.begin(), dh_below_.end(), 0.0f);
                bptt_mat_step(ly.g_uf.data(), ly.uf.data(), df_.data(), ls_below.h_out.data(), dh_below_.data(), H_, H_, avx2_);
                bptt_mat_step(ly.g_ug.data(), ly.ug.data(), dg_.data(), ls_below.h_out.data(), dh_below_.data(), H_, H_, avx2_);
                bptt_mat_step(ly.g_uo.data(), ly.uo.data(), do_.data(), ls_below.h_out.data(), dh_below_.data(), H_, H_, avx2_);
                for (std::size_t i = 0; i < H_; ++i) {
                    ls_below.dh_ext[i] += dh_below_[i];
                }
            }

            // Update sparse bigram/WRT token embedding (layer 0) BEFORE clearing df/dg/do!
            if (l == 0) {
                const std::size_t toff = static_cast<std::size_t>(sc.tok_id) * H_;
                rmsprop_update_sparse(ly.tf.data() + toff, ly.v_tf.data() + toff, df_.data(), lr_tok, H_, avx2_);
                rmsprop_update_sparse(ly.tg.data() + toff, ly.v_tg.data() + toff, dg_.data(), lr_tok, H_, avx2_);
                rmsprop_update_sparse(ly.to.data() + toff, ly.v_to.data() + toff, do_.data(), lr_tok, H_, avx2_);
            }

            const std::size_t eoff = static_cast<std::size_t>(sc.in_byte) * H_;
            adam_update_grad(ly.ef.data() + eoff, ly.m_ef.data() + eoff, ly.v_ef.data() + eoff, df_.data(), lr_emb, H_, avx2_, false);
            adam_update_grad(ly.eg.data() + eoff, ly.m_eg.data() + eoff, ly.v_eg.data() + eoff, dg_.data(), lr_emb, H_, avx2_, false);
            adam_update_grad(ly.eo.data() + eoff, ly.m_eo.data() + eoff, ly.v_eo.data() + eoff, do_.data(), lr_emb, H_, avx2_, false);

            idx = (idx + horizon_ - 1u) % horizon_;
        }

        adam_update_grad(ly.wf.data(), ly.m_wf.data(), ly.v_wf.data(), ly.g_wf.data(), lr_rec, H_ * H_, avx2_);
        adam_update_grad(ly.wg.data(), ly.m_wg.data(), ly.v_wg.data(), ly.g_wg.data(), lr_rec, H_ * H_, avx2_);
        adam_update_grad(ly.wo.data(), ly.m_wo.data(), ly.v_wo.data(), ly.g_wo.data(), lr_rec, H_ * H_, avx2_);
        if (l > 0) {
            adam_update_grad(ly.uf.data(), ly.m_uf.data(), ly.v_uf.data(), ly.g_uf.data(), lr_rec, H_ * H_, avx2_);
            adam_update_grad(ly.ug.data(), ly.m_ug.data(), ly.v_ug.data(), ly.g_ug.data(), lr_rec, H_ * H_, avx2_);
            adam_update_grad(ly.uo.data(), ly.m_uo.data(), ly.v_uo.data(), ly.g_uo.data(), lr_rec, H_ * H_, avx2_);
        }
        adam_update_grad(ly.gf.data(), ly.m_gf.data(), ly.v_gf.data(), ly.g_gf.data(), lr_rec, H_, avx2_);
        adam_update_grad(ly.gg.data(), ly.m_gg.data(), ly.v_gg.data(), ly.g_gg.data(), lr_rec, H_, avx2_);
        adam_update_grad(ly.go.data(), ly.m_go.data(), ly.v_go.data(), ly.g_go.data(), lr_rec, H_, avx2_);
        adam_update_grad(ly.bf.data(), ly.m_bf.data(), ly.v_bf.data(), ly.g_bf.data(), lr_rec, H_, avx2_);
        adam_update_grad(ly.bg.data(), ly.m_bg.data(), ly.v_bg.data(), ly.g_bg.data(), lr_rec, H_, avx2_);
        adam_update_grad(ly.bo.data(), ly.m_bo.data(), ly.v_bo.data(), ly.g_bo.data(), lr_rec, H_, avx2_);
    }
}

void LstmModel::update_byte(std::uint8_t actual_byte, std::uint32_t tok_id) {
    const float decay = 1.0f / std::sqrt(1.0f + static_cast<float>(step_) * (1.0f / 262144.0f));
    const float lr_out = ((step_ < 16384u) ? 0.045f : 0.024f) * decay;

    if (hist_filled_ > 0u) {
        const std::size_t prev_idx = (hist_pos_ + horizon_ - 1u) % horizon_;
        auto& prev_sc = hist_[prev_idx];

        update_output_layer_adam(wout_.data(), m_wout_.data(), v_wout_.data(),
                                 bout_.data(), m_bout_.data(), v_bout_.data(),
                                 h_concat_.data(), probs_.data(), actual_byte,
                                 dh_all_.data(), lr_out, total_H_, avx2_);
        for (std::size_t l = 0; l < num_layers_; ++l) {
            std::memcpy(prev_sc.ls[l].dh_ext.data(), dh_all_.data() + l * H_, H_ * sizeof(float));
        }

        if ((step_ % stride_) == 0u) {
            run_bptt();
        }
    } else {
        for (std::size_t k = 0; k < 256u; ++k) {
            const float dl = probs_[k] - (k == actual_byte ? 1.0f : 0.0f);
            bout_[k] -= 0.05f * dl;
        }
    }

    forward_step(actual_byte, static_cast<std::uint16_t>(tok_id & kTokMask));
    hist_pos_ = (hist_pos_ + 1u) % horizon_;
    if (hist_filled_ < horizon_) ++hist_filled_;
    ++step_;
}

}  // namespace fzip::fzac_v3
