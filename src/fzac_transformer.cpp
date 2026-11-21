// fzip — Induction-Head Online Causal Self-Attention Transformer Predictor
// Combines 4-tap causal depthwise conv, Induction-Head KV alignment
// (K_t = context BEFORE b_t, V_t = observed b_t, Q_t = current context),
// QK-Norm + learned per-head relative position biases over W=64 window,
// SiLU FFN, exact Transformer-XL chunk BPTT, and pure bit-exact AVX2/scalar.
#include "fzac_transformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#define FZAC_TF_X64 1
#else
#define FZAC_TF_X64 0
#endif

namespace fzip::fzac_v3 {

namespace {

constexpr float kAttnScale = 4.0f;  // Cosine-similarity QK-Norm attention temperature

auto detect_avx2_tf() -> bool {
#if FZAC_TF_X64
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

inline auto fast_sigmoid_scalar(float x) -> float {
    return 1.0f / (1.0f + fast_exp_scalar(-x));
}

#if FZAC_TF_X64
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

inline auto dot_exact(const float* a, const float* b, std::size_t n, bool use_avx2) -> float {
#if FZAC_TF_X64
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

void matvec_exact(float* out, const float* M, const float* x,
                  std::size_t rows, std::size_t cols, bool add, bool use_avx2) {
#if FZAC_TF_X64
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
            const float v0 = hsum256_exact(a0);
            const float v1 = hsum256_exact(a1);
            const float v2 = hsum256_exact(a2);
            const float v3 = hsum256_exact(a3);
            out[r + 0u] = add ? (out[r + 0u] + v0) : v0;
            out[r + 1u] = add ? (out[r + 1u] + v1) : v1;
            out[r + 2u] = add ? (out[r + 2u] + v2) : v2;
            out[r + 3u] = add ? (out[r + 3u] + v3) : v3;
        }
        for (; r < rows; ++r) {
            const float* row = M + r * cols;
            __m256 acc = _mm256_setzero_ps();
            for (std::size_t c = 0; c < cols; c += 8u) {
                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(row + c),
                                                       _mm256_loadu_ps(x + c)));
            }
            const float v = hsum256_exact(acc);
            out[r] = add ? (out[r] + v) : v;
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
        const float v = ((acc[0] + acc[4]) + (acc[1] + acc[5])) +
                        ((acc[2] + acc[6]) + (acc[3] + acc[7]));
        out[r] = add ? (out[r] + v) : v;
    }
}

void bptt_linear_step(float* gW, const float* W, const float* dy,
                      const float* x, float* dx,
                      std::size_t rows, std::size_t cols, bool use_avx2) {
#if FZAC_TF_X64
    if (use_avx2) {
        for (std::size_t r = 0; r < rows; ++r) {
            const __m256 vd = _mm256_set1_ps(dy[r]);
            float* grow = gW + r * cols;
            const float* wrow = W + r * cols;
            for (std::size_t c = 0; c < cols; c += 8u) {
                const __m256 vx = _mm256_loadu_ps(x + c);
                _mm256_storeu_ps(grow + c, _mm256_add_ps(_mm256_loadu_ps(grow + c), _mm256_mul_ps(vx, vd)));
                const __m256 vw = _mm256_loadu_ps(wrow + c);
                _mm256_storeu_ps(dx + c, _mm256_add_ps(_mm256_loadu_ps(dx + c), _mm256_mul_ps(vw, vd)));
            }
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t r = 0; r < rows; ++r) {
        const float d = dy[r];
        float* grow = gW + r * cols;
        const float* wrow = W + r * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            grow[c] = grow[c] + (x[c] * d);
            dx[c] = dx[c] + (wrow[c] * d);
        }
    }
}

void rmsnorm_forward(const float* a, const float* gamma, const float* bias,
                     float* u_out, float* y_out, float& inv_rms_out,
                     std::size_t n, bool use_avx2) {
    const float inv_n = 1.0f / static_cast<float>(n);
    const float ss = dot_exact(a, a, n, use_avx2) * inv_n + 1e-4f;
    const float r = 1.0f / std::sqrt(ss);
    inv_rms_out = r;
#if FZAC_TF_X64
    if (use_avx2) {
        const __m256 vr = _mm256_set1_ps(r);
        for (std::size_t i = 0; i < n; i += 8u) {
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
    for (std::size_t i = 0; i < n; ++i) {
        const float u = a[i] * r;
        u_out[i] = u;
        y_out[i] = u * gamma[i] + bias[i];
    }
}

void rmsnorm_backward_inplace(float* d_io, const float* u, const float* gamma,
                              float inv_rms, float* g_gamma, float* g_bias,
                              std::size_t n, bool use_avx2) {
    const float inv_n = 1.0f / static_cast<float>(n);
    float du_tmp[256];
#if FZAC_TF_X64
    if (use_avx2) {
        __m256 vdot = _mm256_setzero_ps();
        for (std::size_t i = 0; i < n; i += 8u) {
            const __m256 vdy = _mm256_loadu_ps(d_io + i);
            const __m256 vu = _mm256_loadu_ps(u + i);
            const __m256 vg = _mm256_loadu_ps(gamma + i);
            _mm256_storeu_ps(g_bias + i, _mm256_add_ps(_mm256_loadu_ps(g_bias + i), vdy));
            _mm256_storeu_ps(g_gamma + i, _mm256_add_ps(_mm256_loadu_ps(g_gamma + i), _mm256_mul_ps(vdy, vu)));
            const __m256 vdu = _mm256_mul_ps(vdy, vg);
            _mm256_storeu_ps(du_tmp + i, vdu);
            vdot = _mm256_add_ps(vdot, _mm256_mul_ps(vdu, vu));
        }
        const float mean_du_u = hsum256_exact(vdot) * inv_n;
        const __m256 vmean = _mm256_set1_ps(mean_du_u);
        const __m256 vinv = _mm256_set1_ps(inv_rms);
        for (std::size_t i = 0; i < n; i += 8u) {
            const __m256 vdu = _mm256_loadu_ps(du_tmp + i);
            const __m256 vu = _mm256_loadu_ps(u + i);
            _mm256_storeu_ps(d_io + i, _mm256_mul_ps(vinv, _mm256_sub_ps(vdu, _mm256_mul_ps(vu, vmean))));
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (std::size_t i = 0; i < n; i += 8u) {
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
         ((acc[2] + acc[6]) + (acc[3] + acc[7]))) * inv_n;
    for (std::size_t i = 0; i < n; ++i) {
        d_io[i] = inv_rms * (du_tmp[i] - u[i] * mean_du_u);
    }
}

void silu_forward(const float* u, float* y, std::size_t n, bool use_avx2) {
#if FZAC_TF_X64
    if (use_avx2) {
        for (std::size_t i = 0; i < n; i += 8u) {
            const __m256 vu = _mm256_loadu_ps(u + i);
            const __m256 vs = fast_sigmoid_avx2(vu);
            _mm256_storeu_ps(y + i, _mm256_mul_ps(vu, vs));
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = u[i] * fast_sigmoid_scalar(u[i]);
    }
}

void silu_backward_inplace(float* d_io, const float* u, std::size_t n, bool use_avx2) {
#if FZAC_TF_X64
    if (use_avx2) {
        const __m256 vone = _mm256_set1_ps(1.0f);
        for (std::size_t i = 0; i < n; i += 8u) {
            const __m256 vu = _mm256_loadu_ps(u + i);
            const __m256 vs = fast_sigmoid_avx2(vu);
            const __m256 deriv = _mm256_mul_ps(vs, _mm256_add_ps(vone, _mm256_mul_ps(vu, _mm256_sub_ps(vone, vs))));
            _mm256_storeu_ps(d_io + i, _mm256_mul_ps(_mm256_loadu_ps(d_io + i), deriv));
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t i = 0; i < n; ++i) {
        const float s = fast_sigmoid_scalar(u[i]);
        d_io[i] = d_io[i] * (s * (1.0f + u[i] * (1.0f - s)));
    }
}

// Pure contiguous AVX2/scalar Multi-Head Causal Attention Forward (ZERO AVX<->SSE transitions!)
// Computes scores[0..att_len-1], vectorized AVX2 softmax (padded to multiple of 8), and V weighted sum.
void attention_forward(const float* q, const float* k_cache, const float* v_cache,
                       const float* rel_b, std::size_t kv_slot, std::size_t kv_cap,
                       std::size_t att_len, std::size_t num_heads, std::size_t dh,
                       std::size_t D, std::size_t window,
                       float* attn_out, float* ctx_out, bool use_avx2) {
    float scores[64];
    const std::size_t pad_len = (att_len + 7u) & ~std::size_t{7};

#if FZAC_TF_X64
    if (use_avx2 && dh == 16u) {
        for (std::size_t h = 0; h < num_heads; ++h) {
            const float* qh = q + h * dh;
            const float* rbh = rel_b + h * window;
            float* ah = attn_out + h * window;
            float* ch = ctx_out + h * dh;

            const __m256 vq0 = _mm256_loadu_ps(qh);
            const __m256 vq1 = _mm256_loadu_ps(qh + 8);
            float max_s = -1e30f;
            for (std::size_t d = 0; d < att_len; ++d) {
                const std::size_t slot = (kv_slot + kv_cap - d) % kv_cap;
                const float* kh = k_cache + slot * D + h * dh;
                const __m256 prod = _mm256_add_ps(
                    _mm256_mul_ps(vq0, _mm256_loadu_ps(kh)),
                    _mm256_mul_ps(vq1, _mm256_loadu_ps(kh + 8)));
                const float s = hsum256_exact(prod) * kAttnScale + rbh[d];
                scores[d] = s;
                if (s > max_s) max_s = s;
            }
            for (std::size_t d = att_len; d < pad_len; ++d) scores[d] = max_s - 30.0f;

            // Vectorized softmax over pad_len (exact match with scalar 8-lane tree!)
            const __m256 vmax = _mm256_set1_ps(max_s);
            __m256 vsum = _mm256_setzero_ps();
            for (std::size_t d = 0; d < pad_len; d += 8u) {
                const __m256 ve = fast_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(scores + d), vmax));
                _mm256_storeu_ps(ah + d, ve);
                // Zero out padding lanes before summing
                if (d + 8u > att_len) {
                    for (std::size_t j = att_len; j < d + 8u; ++j) ah[j] = 0.0f;
                    vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(ah + d));
                } else {
                    vsum = _mm256_add_ps(vsum, ve);
                }
            }
            const float sum_e = hsum256_exact(vsum);
            const __m256 vinv = _mm256_set1_ps(1.0f / sum_e);
            for (std::size_t d = 0; d < pad_len; d += 8u) {
                _mm256_storeu_ps(ah + d, _mm256_mul_ps(_mm256_loadu_ps(ah + d), vinv));
            }

            // Context weighted sum in registers
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            for (std::size_t d = 0; d < att_len; ++d) {
                const std::size_t slot = (kv_slot + kv_cap - d) % kv_cap;
                const float* vh = v_cache + slot * D + h * dh;
                const __m256 va = _mm256_set1_ps(ah[d]);
                c0 = _mm256_add_ps(c0, _mm256_mul_ps(va, _mm256_loadu_ps(vh)));
                c1 = _mm256_add_ps(c1, _mm256_mul_ps(va, _mm256_loadu_ps(vh + 8)));
            }
            _mm256_storeu_ps(ch, c0);
            _mm256_storeu_ps(ch + 8, c1);
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t h = 0; h < num_heads; ++h) {
        const float* qh = q + h * dh;
        const float* rbh = rel_b + h * window;
        float* ah = attn_out + h * window;
        float* ch = ctx_out + h * dh;

        float max_s = -1e30f;
        for (std::size_t d = 0; d < att_len; ++d) {
            const std::size_t slot = (kv_slot + kv_cap - d) % kv_cap;
            const float* kh = k_cache + slot * D + h * dh;
            float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (std::size_t c = 0; c < dh; c += 8u) {
                for (std::size_t j = 0; j < 8u; ++j) {
                    acc[j] = acc[j] + (qh[c + j] * kh[c + j]);
                }
            }
            const float dp = ((acc[0] + acc[4]) + (acc[1] + acc[5])) +
                             ((acc[2] + acc[6]) + (acc[3] + acc[7]));
            const float s = dp * kAttnScale + rbh[d];
            scores[d] = s;
            if (s > max_s) max_s = s;
        }
        for (std::size_t d = att_len; d < pad_len; ++d) scores[d] = max_s - 30.0f;

        float sacc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (std::size_t d = 0; d < pad_len; d += 8u) {
            for (std::size_t j = 0; j < 8u; ++j) {
                const float e = (d + j < att_len) ? fast_exp_scalar(scores[d + j] - max_s) : 0.0f;
                ah[d + j] = e;
                sacc[j] = sacc[j] + e;
            }
        }
        const float sum_e = ((sacc[0] + sacc[4]) + (sacc[1] + sacc[5])) +
                            ((sacc[2] + sacc[6]) + (sacc[3] + sacc[7]));
        const float inv_sum = 1.0f / sum_e;
        for (std::size_t d = 0; d < pad_len; ++d) {
            ah[d] = ah[d] * inv_sum;
        }

        std::memset(ch, 0, dh * sizeof(float));
        for (std::size_t d = 0; d < att_len; ++d) {
            const std::size_t slot = (kv_slot + kv_cap - d) % kv_cap;
            const float* vh = v_cache + slot * D + h * dh;
            const float w = ah[d];
            for (std::size_t j = 0; j < dh; ++j) {
                ch[j] = ch[j] + (w * vh[j]);
            }
        }
    }
}

// Pure contiguous AVX2/scalar Multi-Head Causal Attention Backward (ZERO AVX<->SSE transitions!)
void attention_backward(const float* d_ctx, const float* q, const float* attn,
                        const float* k_cache, const float* v_cache,
                        std::size_t t, std::size_t kv_slot, std::size_t kv_cap,
                        std::size_t att_len, std::size_t num_heads, std::size_t dh,
                        std::size_t D, std::size_t window,
                        float* dq_out, float* dk_chunk, float* dv_chunk,
                        float* g_rel_b, bool use_avx2) {
    float d_attn[64];

#if FZAC_TF_X64
    if (use_avx2 && dh == 16u) {
        for (std::size_t h = 0; h < num_heads; ++h) {
            const float* dch = d_ctx + h * dh;
            const float* qh = q + h * dh;
            const float* ah = attn + h * window;
            float* dqh = dq_out + h * dh;
            float* grbh = g_rel_b + h * window;

            const __m256 vdc0 = _mm256_loadu_ps(dch);
            const __m256 vdc1 = _mm256_loadu_ps(dch + 8);
            const __m256 vq0 = _mm256_loadu_ps(qh);
            const __m256 vq1 = _mm256_loadu_ps(qh + 8);

            float dot_da = 0.0f;
            for (std::size_t d = 0; d < att_len; ++d) {
                const std::size_t slot = (kv_slot + kv_cap - d) % kv_cap;
                const float* vh = v_cache + slot * D + h * dh;
                const __m256 prod = _mm256_add_ps(
                    _mm256_mul_ps(vdc0, _mm256_loadu_ps(vh)),
                    _mm256_mul_ps(vdc1, _mm256_loadu_ps(vh + 8)));
                const float da = hsum256_exact(prod);
                d_attn[d] = da;
                dot_da = dot_da + (ah[d] * da);
            }

            __m256 vdq0 = _mm256_setzero_ps();
            __m256 vdq1 = _mm256_setzero_ps();
            for (std::size_t d = 0; d < att_len; ++d) {
                const std::size_t slot = (kv_slot + kv_cap - d) % kv_cap;
                const float* kh = k_cache + slot * D + h * dh;
                const float w = ah[d];
                const float ds = w * (d_attn[d] - dot_da);
                grbh[d] = grbh[d] + ds;
                const __m256 vsds = _mm256_set1_ps(ds * kAttnScale);

                vdq0 = _mm256_add_ps(vdq0, _mm256_mul_ps(vsds, _mm256_loadu_ps(kh)));
                vdq1 = _mm256_add_ps(vdq1, _mm256_mul_ps(vsds, _mm256_loadu_ps(kh + 8)));

                if (d <= t) {
                    const std::size_t jc = t - d;
                    const __m256 vw = _mm256_set1_ps(w);
                    float* dvh = dv_chunk + jc * D + h * dh;
                    _mm256_storeu_ps(dvh,     _mm256_add_ps(_mm256_loadu_ps(dvh),     _mm256_mul_ps(vw, vdc0)));
                    _mm256_storeu_ps(dvh + 8, _mm256_add_ps(_mm256_loadu_ps(dvh + 8), _mm256_mul_ps(vw, vdc1)));

                    float* dkh = dk_chunk + jc * D + h * dh;
                    _mm256_storeu_ps(dkh,     _mm256_add_ps(_mm256_loadu_ps(dkh),     _mm256_mul_ps(vsds, vq0)));
                    _mm256_storeu_ps(dkh + 8, _mm256_add_ps(_mm256_loadu_ps(dkh + 8), _mm256_mul_ps(vsds, vq1)));
                }
            }
            _mm256_storeu_ps(dqh, vdq0);
            _mm256_storeu_ps(dqh + 8, vdq1);
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t h = 0; h < num_heads; ++h) {
        const float* dch = d_ctx + h * dh;
        const float* qh = q + h * dh;
        const float* ah = attn + h * window;
        float* dqh = dq_out + h * dh;
        float* grbh = g_rel_b + h * window;

        float dot_da = 0.0f;
        for (std::size_t d = 0; d < att_len; ++d) {
            const std::size_t slot = (kv_slot + kv_cap - d) % kv_cap;
            const float* vh = v_cache + slot * D + h * dh;
            float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (std::size_t c = 0; c < dh; c += 8u) {
                for (std::size_t j = 0; j < 8u; ++j) {
                    acc[j] = acc[j] + (dch[c + j] * vh[c + j]);
                }
            }
            const float da = ((acc[0] + acc[4]) + (acc[1] + acc[5])) +
                             ((acc[2] + acc[6]) + (acc[3] + acc[7]));
            d_attn[d] = da;
            dot_da = dot_da + (ah[d] * da);
        }

        std::memset(dqh, 0, dh * sizeof(float));
        for (std::size_t d = 0; d < att_len; ++d) {
            const std::size_t slot = (kv_slot + kv_cap - d) % kv_cap;
            const float* kh = k_cache + slot * D + h * dh;
            const float w = ah[d];
            const float ds = w * (d_attn[d] - dot_da);
            grbh[d] = grbh[d] + ds;
            const float sds = ds * kAttnScale;
            for (std::size_t j = 0; j < dh; ++j) {
                dqh[j] = dqh[j] + (sds * kh[j]);
            }
            if (d <= t) {
                const std::size_t jc = t - d;
                float* dvh = dv_chunk + jc * D + h * dh;
                float* dkh = dk_chunk + jc * D + h * dh;
                for (std::size_t j = 0; j < dh; ++j) {
                    dvh[j] = dvh[j] + (w * dch[j]);
                    dkh[j] = dkh[j] + (sds * qh[j]);
                }
            }
        }
    }
}

void softmax_256(const float* logits, float* probs, bool use_avx2) {
    float max_l = logits[0];
    for (std::size_t k = 1u; k < 256u; ++k) {
        if (logits[k] > max_l) max_l = logits[k];
    }
    float sum_e = 0.0f;
#if FZAC_TF_X64
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

void update_output_layer_adam(float* wout, float* mwout, float* vwout,
                              float* bout, float* mbout, float* vbout,
                              const float* h_out, const float* probs,
                              std::uint8_t actual_byte, float* dh_out,
                              float lr, std::size_t D, bool use_avx2) {
    constexpr float kB1 = 0.50f, k1mB1 = 0.50f;
    constexpr float kB2 = 0.99f, k1mB2 = 0.01f;
    constexpr float kEps = 1e-4f;
    std::memset(dh_out, 0, D * sizeof(float));

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

    for (std::size_t idx = 0; idx < n_act; ++idx) {
        const std::size_t k = active_k[idx];
        const float g = std::clamp(active_dl[idx], -2.0f, 2.0f);
        const float mm = mbout[k] * kB1 + g * k1mB1;
        const float vv = vbout[k] * kB2 + (g * g) * k1mB2;
        mbout[k] = mm;
        vbout[k] = vv;
        bout[k] = std::clamp(bout[k] - (lr * mm) / (std::sqrt(vv) + kEps), -8.0f, 8.0f);
    }

#if FZAC_TF_X64
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
            float* wrow = wout + k * D;
            float* mrow = mwout + k * D;
            float* vrow = vwout + k * D;
            for (std::size_t c = 0; c < D; c += 8u) {
                const __m256 vw = _mm256_loadu_ps(wrow + c);
                const __m256 vdh = _mm256_loadu_ps(dh_out + c);
                _mm256_storeu_ps(dh_out + c, _mm256_add_ps(vdh, _mm256_mul_ps(vw, vdl)));

                const __m256 vh = _mm256_loadu_ps(h_out + c);
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
        float* wrow = wout + k * D;
        float* mrow = mwout + k * D;
        float* vrow = vwout + k * D;
        for (std::size_t c = 0; c < D; ++c) {
            dh_out[c] = dh_out[c] + (wrow[c] * dl);
            const float g = std::clamp(h_out[c] * dl, -2.0f, 2.0f);
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
#if FZAC_TF_X64
    if (use_avx2 && (n & 7u) == 0u) {
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
#if FZAC_TF_X64
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

struct XorShift32Tf {
    std::uint32_t s = 0x517CC1B7u;
    auto next_f() -> float {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return static_cast<float>(static_cast<std::int32_t>(s)) * (1.0f / 2147483648.0f);
    }
};

}  // namespace

TransformerModel::TransformerModel(std::size_t d_model, std::size_t num_heads,
                                   std::size_t window, std::size_t chunk_len)
    : D_(std::clamp<std::size_t>((d_model + 7u) & ~std::size_t{7}, 32u, 128u)),
      num_heads_(std::clamp<std::size_t>(num_heads, 1u, 8u)),
      dh_(D_ / num_heads_),
      ffn_dim_(D_ * 2u),
      window_(std::clamp<std::size_t>((window + 7u) & ~std::size_t{7}, 16u, 64u)),
      chunk_len_(std::clamp<std::size_t>(chunk_len, 8u, 32u)),
      kv_cap_(window_ + chunk_len_),
      avx2_(detect_avx2_tf()) {
    XorShift32Tf rng;
    const float r_emb = 0.25f;
    const float r_d = 1.0f / std::sqrt(static_cast<float>(D_));
    const float r_ffn = 1.0f / std::sqrt(static_cast<float>(ffn_dim_));

    e_byte_.resize(256u * D_);
    m_ebyte_.assign(256u * D_, 0.0f);
    v_ebyte_.assign(256u * D_, 1e-3f);
    for (float& x : e_byte_) x = rng.next_f() * r_emb;

    e_tok_.assign(kTokBuckets * D_, 0.0f);
    v_etok_.assign(kTokBuckets * D_, 1e-3f);

    // 4-tap depthwise conv initialized for crisp multi-byte context
    conv_w_.assign(kConvK * D_, 0.0f);
    for (std::size_t i = 0; i < D_; ++i) {
        conv_w_[0u * D_ + i] = 0.85f + 0.10f * rng.next_f();
        conv_w_[1u * D_ + i] = 0.35f + 0.10f * rng.next_f();
        conv_w_[2u * D_ + i] = 0.15f + 0.08f * rng.next_f();
        conv_w_[3u * D_ + i] = 0.05f + 0.05f * rng.next_f();
    }
    m_convw_.assign(kConvK * D_, 0.0f);
    v_convw_.assign(kConvK * D_, 1e-3f);
    g_convw_.assign(kConvK * D_, 0.0f);
    conv_b_.assign(D_, 0.0f);
    m_convb_.assign(D_, 0.0f);
    v_convb_.assign(D_, 1e-3f);
    g_convb_.assign(D_, 0.0f);
    e_ring_.assign(kConvK * D_, 0.0f);
    x_prev_live_.assign(D_, 0.0f);

    auto init_norm = [&](std::vector<float>& g, std::vector<float>& mg, std::vector<float>& vg, std::vector<float>& gg,
                         std::vector<float>& b, std::vector<float>& mb, std::vector<float>& vb, std::vector<float>& gb) {
        g.assign(D_, 1.0f); mg.assign(D_, 0.0f); vg.assign(D_, 1e-3f); gg.assign(D_, 0.0f);
        b.assign(D_, 0.0f); mb.assign(D_, 0.0f); vb.assign(D_, 1e-3f); gb.assign(D_, 0.0f);
    };
    init_norm(gx_, m_gx_, v_gx_, g_gx_, bx_, m_bx_, v_bx_, g_bx_);
    init_norm(g1_, m_g1_, v_g1_, grad_g1_, b1_, m_b1_, v_b1_, grad_b1_);
    init_norm(g2_, m_g2_, v_g2_, grad_g2_, b2_, m_b2_, v_b2_, grad_b2_);

    const std::size_t dd = D_ * D_;
    wq_.resize(dd); wk_.resize(dd); wv_.resize(dd); wo_.resize(dd);
    for (std::size_t i = 0; i < dd; ++i) {
        wq_[i] = rng.next_f() * r_d;
        wk_[i] = rng.next_f() * r_d;
        wv_[i] = rng.next_f() * r_d;
        wo_[i] = rng.next_f() * (r_d * 0.6f);
    }
    // Seed W_Q near W_K so Induction-Head cosine matching fires from byte 1!
    for (std::size_t i = 0; i < dd; ++i) {
        wq_[i] = 0.80f * wk_[i] + 0.20f * wq_[i];
    }
    m_wq_.assign(dd, 0.0f); v_wq_.assign(dd, 1e-3f); g_wq_.assign(dd, 0.0f);
    m_wk_.assign(dd, 0.0f); v_wk_.assign(dd, 1e-3f); g_wk_.assign(dd, 0.0f);
    m_wv_.assign(dd, 0.0f); v_wv_.assign(dd, 1e-3f); g_wv_.assign(dd, 0.0f);
    m_wo_.assign(dd, 0.0f); v_wo_.assign(dd, 1e-3f); g_wo_.assign(dd, 0.0f);

    // Multi-scale relative position biases: local -> medium -> content-pure across W=64!
    rel_b_.assign(num_heads_ * window_, 0.0f);
    const float slopes[8] = {0.20f, 0.07f, 0.02f, 0.002f, 0.15f, 0.05f, 0.01f, 0.001f};
    for (std::size_t h = 0; h < num_heads_; ++h) {
        const float sl = slopes[h & 7u];
        for (std::size_t d = 0; d < window_; ++d) {
            rel_b_[h * window_ + d] = -sl * static_cast<float>(d);
        }
    }
    m_relb_.assign(num_heads_ * window_, 0.0f);
    v_relb_.assign(num_heads_ * window_, 1e-3f);
    g_relb_.assign(num_heads_ * window_, 0.0f);

    w1_.resize(ffn_dim_ * D_);
    for (float& x : w1_) x = rng.next_f() * r_d;
    m_w1_.assign(ffn_dim_ * D_, 0.0f); v_w1_.assign(ffn_dim_ * D_, 1e-3f); g_w1_.assign(ffn_dim_ * D_, 0.0f);
    bf1_.assign(ffn_dim_, 0.0f); m_bf1_.assign(ffn_dim_, 0.0f); v_bf1_.assign(ffn_dim_, 1e-3f); g_bf1_.assign(ffn_dim_, 0.0f);

    w2_.resize(D_ * ffn_dim_);
    for (float& x : w2_) x = rng.next_f() * (r_ffn * 0.5f);
    m_w2_.assign(D_ * ffn_dim_, 0.0f); v_w2_.assign(D_ * ffn_dim_, 1e-3f); g_w2_.assign(D_ * ffn_dim_, 0.0f);
    bf2_.assign(D_, 0.0f); m_bf2_.assign(D_, 0.0f); v_bf2_.assign(D_, 1e-3f); g_bf2_.assign(D_, 0.0f);

    k_cache_.assign(kv_cap_ * D_, 0.0f);
    v_cache_.assign(kv_cap_ * D_, 0.0f);

    wout_.assign(256u * D_, 0.0f);
    m_wout_.assign(256u * D_, 0.0f);
    v_wout_.assign(256u * D_, 1e-3f);
    bout_.assign(256u, 0.0f);
    m_bout_.assign(256u, 0.0f);
    v_bout_.assign(256u, 1e-3f);
    h_out_.assign(D_, 0.0f);
    probs_.fill(1.0f / 256.0f);

    chunk_.resize(chunk_len_);
    for (auto& sc : chunk_) {
        sc.conv_e.assign(kConvK * D_, 0.0f);
        sc.conv_u.assign(D_, 0.0f);
        sc.x_un.assign(D_, 0.0f);
        sc.x_hat.assign(D_, 0.0f);
        sc.x_prev.assign(D_, 0.0f);
        sc.q.assign(D_, 0.0f);
        sc.inv_nq.assign(num_heads_, 1.0f);
        sc.inv_nk.assign(num_heads_, 1.0f);
        sc.attn.assign(num_heads_ * window_, 0.0f);
        sc.ctx.assign(D_, 0.0f);
        sc.x1_un.assign(D_, 0.0f);
        sc.x1_norm.assign(D_, 0.0f);
        sc.ffn_u.assign(ffn_dim_, 0.0f);
        sc.ffn_h.assign(ffn_dim_, 0.0f);
        sc.x2_un.assign(D_, 0.0f);
        sc.dx_ext.assign(D_, 0.0f);
    }

    dk_chunk_.assign(chunk_len_ * D_, 0.0f);
    dv_chunk_.assign(chunk_len_ * D_, 0.0f);
    dx2_.assign(D_, 0.0f);
    dx1_.assign(D_, 0.0f);
    dx_hat_.assign(D_, 0.0f);
    dx_from_next_k_.assign(D_, 0.0f);
    d_ctx_.assign(D_, 0.0f);
    dq_.assign(D_, 0.0f);
    dk_raw_.assign(D_, 0.0f);
    d_ffn_h_.assign(ffn_dim_, 0.0f);
    d_ffn_u_.assign(ffn_dim_, 0.0f);
    de_shift_.assign(kConvK * D_, 0.0f);

    build_tree();
}

void TransformerModel::build_tree() {
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

auto TransformerModel::predict_bit(std::uint32_t c0) const -> std::uint32_t {
    const std::uint32_t tot = tree_[c0];
    const std::uint32_t p1 = tree_[c0 * 2u + 1u];
    const std::uint64_t p = (static_cast<std::uint64_t>(p1) * 4096u + (tot >> 1)) / tot;
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(p, 1u, 4095u));
}

void TransformerModel::forward_step(std::uint8_t in_byte, std::uint16_t tok_id) {
    auto& sc = chunk_[chunk_pos_];
    sc.in_byte = in_byte;
    sc.tok_id = tok_id;
    sc.kv_slot = kv_pos_;
    const std::size_t att_len = std::min(kv_filled_ + 1u, window_);
    sc.att_len = att_len;
    std::memcpy(sc.x_prev.data(), x_prev_live_.data(), D_ * sizeof(float));

    // 1. Shift 4-tap embedding history & form e_t = E_byte[in_byte] + E_tok[tok_id]
    for (std::size_t m = kConvK - 1u; m >= 1u; --m) {
        std::memcpy(e_ring_.data() + m * D_, e_ring_.data() + (m - 1u) * D_, D_ * sizeof(float));
    }
    const float* eb = e_byte_.data() + static_cast<std::size_t>(in_byte) * D_;
    const float* et = e_tok_.data() + static_cast<std::size_t>(tok_id) * D_;
    for (std::size_t i = 0; i < D_; ++i) {
        e_ring_[i] = eb[i] + et[i];
    }
    std::memcpy(sc.conv_e.data(), e_ring_.data(), kConvK * D_ * sizeof(float));

    // 2. Depthwise causal 4-tap conv -> SiLU -> RMSNorm -> x_hat (current 4-byte context!)
    for (std::size_t i = 0; i < D_; ++i) {
        float acc = conv_b_[i];
        for (std::size_t m = 0; m < kConvK; ++m) {
            acc = acc + (conv_w_[m * D_ + i] * e_ring_[m * D_ + i]);
        }
        sc.conv_u[i] = acc;
    }
    float x_silu[128];
    silu_forward(sc.conv_u.data(), x_silu, D_, avx2_);
    rmsnorm_forward(x_silu, gx_.data(), bx_.data(), sc.x_un.data(), sc.x_hat.data(), sc.inv_rms_x, D_, avx2_);

    // 3. Induction-Head KV Alignment!
    //    K_t = QKNorm(W_K * x_{t-1})  (4-byte context BEFORE b_t!)
    //    V_t = W_V * e_t              (direct representation of observed b_t!)
    //    Q_t = QKNorm(W_Q * x_t)      (current 4-byte context to predict b_{t+1}!)
    float q_raw[128];
    float k_raw[128];
    matvec_exact(q_raw, wq_.data(), sc.x_hat.data(),  D_, D_, false, avx2_);
    matvec_exact(k_raw, wk_.data(), sc.x_prev.data(), D_, D_, false, avx2_);
    float* k_slot = k_cache_.data() + kv_pos_ * D_;
    float* v_slot = v_cache_.data() + kv_pos_ * D_;
    matvec_exact(v_slot, wv_.data(), e_ring_.data(),  D_, D_, false, avx2_);

    for (std::size_t h = 0; h < num_heads_; ++h) {
        const float* qr = q_raw + h * dh_;
        const float* kr = k_raw + h * dh_;
        const float inq = 1.0f / std::sqrt(dot_exact(qr, qr, dh_, avx2_) + 1e-4f);
        const float ink = 1.0f / std::sqrt(dot_exact(kr, kr, dh_, avx2_) + 1e-4f);
        sc.inv_nq[h] = inq;
        sc.inv_nk[h] = ink;
        float* qh = sc.q.data() + h * dh_;
        float* kh = k_slot + h * dh_;
        for (std::size_t j = 0; j < dh_; ++j) {
            qh[j] = qr[j] * inq;
            kh[j] = kr[j] * ink;
        }
    }

    // 4. Causal sliding-window self-attention + residual x1 = x_hat + W_O * ctx
    attention_forward(sc.q.data(), k_cache_.data(), v_cache_.data(), rel_b_.data(),
                      kv_pos_, kv_cap_, att_len, num_heads_, dh_, D_, window_,
                      sc.attn.data(), sc.ctx.data(), avx2_);
    float x1[128];
    std::memcpy(x1, sc.x_hat.data(), D_ * sizeof(float));
    matvec_exact(x1, wo_.data(), sc.ctx.data(), D_, D_, true, avx2_);

    // 5. Pre-FFN RMSNorm + SiLU FFN + residual x2 = x1 + W_2 * silu(W_1 * x1_norm + bf1) + bf2
    rmsnorm_forward(x1, g1_.data(), b1_.data(), sc.x1_un.data(), sc.x1_norm.data(), sc.inv_rms1, D_, avx2_);
    std::memcpy(sc.ffn_u.data(), bf1_.data(), ffn_dim_ * sizeof(float));
    matvec_exact(sc.ffn_u.data(), w1_.data(), sc.x1_norm.data(), ffn_dim_, D_, true, avx2_);
    silu_forward(sc.ffn_u.data(), sc.ffn_h.data(), ffn_dim_, avx2_);

    float x2[128];
    for (std::size_t i = 0; i < D_; ++i) x2[i] = x1[i] + bf2_[i];
    matvec_exact(x2, w2_.data(), sc.ffn_h.data(), D_, ffn_dim_, true, avx2_);

    // 6. Output RMSNorm -> 256-way softmax
    rmsnorm_forward(x2, g2_.data(), b2_.data(), sc.x2_un.data(), h_out_.data(), sc.inv_rms2, D_, avx2_);

    std::array<float, 256> logits{};
    std::memcpy(logits.data(), bout_.data(), 256u * sizeof(float));
    matvec_exact(logits.data(), wout_.data(), h_out_.data(), 256u, D_, true, avx2_);
    softmax_256(logits.data(), probs_.data(), avx2_);
    build_tree();

    // Advance live x_prev & KV cache
    std::memcpy(x_prev_live_.data(), sc.x_hat.data(), D_ * sizeof(float));
    kv_pos_ = (kv_pos_ + 1u) % kv_cap_;
    if (kv_filled_ < window_) ++kv_filled_;
}

void TransformerModel::run_chunk_bptt() {
    const std::size_t steps = chunk_filled_;
    const float decay = 1.0f / std::sqrt(1.0f + static_cast<float>(step_) * (1.0f / 262144.0f));
    const float lr_emb  = ((step_ < 16384u) ? 0.040f : 0.022f) * decay;
    const float lr_tok  = ((step_ < 16384u) ? 0.048f : 0.026f) * decay;
    const float lr_main = ((step_ < 16384u) ? 0.024f : 0.013f) * decay;

    std::fill(dk_chunk_.begin(), dk_chunk_.end(), 0.0f);
    std::fill(dv_chunk_.begin(), dv_chunk_.end(), 0.0f);
    std::fill(dx_from_next_k_.begin(), dx_from_next_k_.end(), 0.0f);
    std::fill(de_shift_.begin(), de_shift_.end(), 0.0f);

    for (std::size_t s = 0; s < steps; ++s) {
        const std::size_t t = steps - 1u - s;
        const auto& sc = chunk_[t];

        // 1. Output RMSNorm backward
        std::memcpy(dx2_.data(), sc.dx_ext.data(), D_ * sizeof(float));
        rmsnorm_backward_inplace(dx2_.data(), sc.x2_un.data(), g2_.data(), sc.inv_rms2,
                                 grad_g2_.data(), grad_b2_.data(), D_, avx2_);

        // 2. FFN residual backward
        std::memcpy(dx1_.data(), dx2_.data(), D_ * sizeof(float));
        for (std::size_t i = 0; i < D_; ++i) g_bf2_[i] = g_bf2_[i] + dx2_[i];
        std::fill(d_ffn_h_.begin(), d_ffn_h_.end(), 0.0f);
        bptt_linear_step(g_w2_.data(), w2_.data(), dx2_.data(), sc.ffn_h.data(), d_ffn_h_.data(), D_, ffn_dim_, avx2_);

        std::memcpy(d_ffn_u_.data(), d_ffn_h_.data(), ffn_dim_ * sizeof(float));
        silu_backward_inplace(d_ffn_u_.data(), sc.ffn_u.data(), ffn_dim_, avx2_);
        for (std::size_t i = 0; i < ffn_dim_; ++i) g_bf1_[i] = g_bf1_[i] + d_ffn_u_[i];
        float dx1_norm[128] = {};
        bptt_linear_step(g_w1_.data(), w1_.data(), d_ffn_u_.data(), sc.x1_norm.data(), dx1_norm, ffn_dim_, D_, avx2_);

        rmsnorm_backward_inplace(dx1_norm, sc.x1_un.data(), g1_.data(), sc.inv_rms1,
                                 grad_g1_.data(), grad_b1_.data(), D_, avx2_);
        for (std::size_t i = 0; i < D_; ++i) {
            dx1_[i] = std::clamp(dx1_[i] + dx1_norm[i], -2.0f, 2.0f);
        }

        // 3. Attention residual backward: x1 = x_hat + W_O * ctx
        for (std::size_t i = 0; i < D_; ++i) {
            dx_hat_[i] = dx1_[i] + dx_from_next_k_[i];
        }
        std::fill(d_ctx_.begin(), d_ctx_.end(), 0.0f);
        bptt_linear_step(g_wo_.data(), wo_.data(), dx1_.data(), sc.ctx.data(), d_ctx_.data(), D_, D_, avx2_);

        // 4. Causal self-attention backward
        attention_backward(d_ctx_.data(), sc.q.data(), sc.attn.data(),
                           k_cache_.data(), v_cache_.data(),
                           t, sc.kv_slot, kv_cap_, sc.att_len, num_heads_, dh_, D_, window_,
                           dq_.data(), dk_chunk_.data(), dv_chunk_.data(), g_relb_.data(), avx2_);

        // 5. QK-Norm L2 backward for Q_t and K_t
        const float* k_slot = k_cache_.data() + sc.kv_slot * D_;
        const float* dkt = dk_chunk_.data() + t * D_;
        float dq_raw[128];
        for (std::size_t h = 0; h < num_heads_; ++h) {
            const float* qh = sc.q.data() + h * dh_;
            const float* kh = k_slot + h * dh_;
            const float* dqh = dq_.data() + h * dh_;
            const float* dkh = dkt + h * dh_;
            const float dot_q = dot_exact(dqh, qh, dh_, avx2_);
            const float dot_k = dot_exact(dkh, kh, dh_, avx2_);
            const float inq = sc.inv_nq[h];
            const float ink = sc.inv_nk[h];
            for (std::size_t j = 0; j < dh_; ++j) {
                dq_raw[h * dh_ + j] = std::clamp(inq * (dqh[j] - qh[j] * dot_q), -1.5f, 1.5f);
                dk_raw_[h * dh_ + j] = std::clamp(ink * (dkh[j] - kh[j] * dot_k), -1.5f, 1.5f);
            }
        }

        // 6. Linear projections backward:
        //    Q_t = W_Q * x_hat   -> adds to dx_hat_
        //    K_t = W_K * x_prev  -> sets dx_from_next_k_ for step t-1!
        //    V_t = W_V * e_t     -> adds directly to de_shift_[0] (e_t)!
        bptt_linear_step(g_wq_.data(), wq_.data(), dq_raw, sc.x_hat.data(), dx_hat_.data(), D_, D_, avx2_);
        std::fill(dx_from_next_k_.begin(), dx_from_next_k_.end(), 0.0f);
        bptt_linear_step(g_wk_.data(), wk_.data(), dk_raw_.data(), sc.x_prev.data(), dx_from_next_k_.data(), D_, D_, avx2_);
        const float* dvt = dv_chunk_.data() + t * D_;
        bptt_linear_step(g_wv_.data(), wv_.data(), dvt, sc.conv_e.data(), de_shift_.data(), D_, D_, avx2_);

        // 7. Input RMSNorm -> SiLU -> 4-tap depthwise conv backward
        float du_conv[128];
        std::memcpy(du_conv, dx_hat_.data(), D_ * sizeof(float));
        rmsnorm_backward_inplace(du_conv, sc.x_un.data(), gx_.data(), sc.inv_rms_x,
                                 g_gx_.data(), g_bx_.data(), D_, avx2_);
        silu_backward_inplace(du_conv, sc.conv_u.data(), D_, avx2_);

        for (std::size_t i = 0; i < D_; ++i) {
            const float du = std::clamp(du_conv[i], -1.5f, 1.5f);
            g_convb_[i] = g_convb_[i] + du;
            for (std::size_t m = 0; m < kConvK; ++m) {
                g_convw_[m * D_ + i] = g_convw_[m * D_ + i] + (du * sc.conv_e[m * D_ + i]);
                de_shift_[m * D_ + i] = de_shift_[m * D_ + i] + (du * conv_w_[m * D_ + i]);
            }
        }

        // 8. Apply accumulated embedding gradient de_shift_[0] to E_byte & E_tok
        const std::size_t toff = static_cast<std::size_t>(sc.tok_id) * D_;
        rmsprop_update_sparse(e_tok_.data() + toff, v_etok_.data() + toff, de_shift_.data(), lr_tok, D_, avx2_);

        const std::size_t eoff = static_cast<std::size_t>(sc.in_byte) * D_;
        adam_update_grad(e_byte_.data() + eoff, m_ebyte_.data() + eoff, v_ebyte_.data() + eoff,
                         de_shift_.data(), lr_emb, D_, avx2_, false);

        for (std::size_t m = 0; m + 1u < kConvK; ++m) {
            std::memcpy(de_shift_.data() + m * D_, de_shift_.data() + (m + 1u) * D_, D_ * sizeof(float));
        }
        std::memset(de_shift_.data() + (kConvK - 1u) * D_, 0, D_ * sizeof(float));
    }

    // Apply Adam updates to all dense Transformer weights
    adam_update_grad(wq_.data(), m_wq_.data(), v_wq_.data(), g_wq_.data(), lr_main, D_ * D_, avx2_);
    adam_update_grad(wk_.data(), m_wk_.data(), v_wk_.data(), g_wk_.data(), lr_main, D_ * D_, avx2_);
    adam_update_grad(wv_.data(), m_wv_.data(), v_wv_.data(), g_wv_.data(), lr_main, D_ * D_, avx2_);
    adam_update_grad(wo_.data(), m_wo_.data(), v_wo_.data(), g_wo_.data(), lr_main, D_ * D_, avx2_);
    adam_update_grad(rel_b_.data(), m_relb_.data(), v_relb_.data(), g_relb_.data(), lr_main, num_heads_ * window_, avx2_);
    adam_update_grad(w1_.data(), m_w1_.data(), v_w1_.data(), g_w1_.data(), lr_main, ffn_dim_ * D_, avx2_);
    adam_update_grad(bf1_.data(), m_bf1_.data(), v_bf1_.data(), g_bf1_.data(), lr_main, ffn_dim_, avx2_);
    adam_update_grad(w2_.data(), m_w2_.data(), v_w2_.data(), g_w2_.data(), lr_main, D_ * ffn_dim_, avx2_);
    adam_update_grad(bf2_.data(), m_bf2_.data(), v_bf2_.data(), g_bf2_.data(), lr_main, D_, avx2_);
    adam_update_grad(conv_w_.data(), m_convw_.data(), v_convw_.data(), g_convw_.data(), lr_main, kConvK * D_, avx2_);
    adam_update_grad(conv_b_.data(), m_convb_.data(), v_convb_.data(), g_convb_.data(), lr_main, D_, avx2_);
    adam_update_grad(gx_.data(), m_gx_.data(), v_gx_.data(), g_gx_.data(), lr_main, D_, avx2_);
    adam_update_grad(bx_.data(), m_bx_.data(), v_bx_.data(), g_bx_.data(), lr_main, D_, avx2_);
    adam_update_grad(g1_.data(), m_g1_.data(), v_g1_.data(), grad_g1_.data(), lr_main, D_, avx2_);
    adam_update_grad(b1_.data(), m_b1_.data(), v_b1_.data(), grad_b1_.data(), lr_main, D_, avx2_);
    adam_update_grad(g2_.data(), m_g2_.data(), v_g2_.data(), grad_g2_.data(), lr_main, D_, avx2_);
    adam_update_grad(b2_.data(), m_b2_.data(), v_b2_.data(), grad_b2_.data(), lr_main, D_, avx2_);
}

void TransformerModel::update_byte(std::uint8_t actual_byte, std::uint32_t tok_id) {
    const float decay = 1.0f / std::sqrt(1.0f + static_cast<float>(step_) * (1.0f / 262144.0f));
    const float lr_out = ((step_ < 16384u) ? 0.045f : 0.024f) * decay;

    if (chunk_filled_ > 0u) {
        auto& prev_sc = chunk_[chunk_filled_ - 1u];
        update_output_layer_adam(wout_.data(), m_wout_.data(), v_wout_.data(),
                                 bout_.data(), m_bout_.data(), v_bout_.data(),
                                 h_out_.data(), probs_.data(), actual_byte,
                                 prev_sc.dx_ext.data(), lr_out, D_, avx2_);

        if (chunk_filled_ == chunk_len_) {
            run_chunk_bptt();
            chunk_pos_ = 0u;
            chunk_filled_ = 0u;
        }
    } else {
        for (std::size_t k = 0; k < 256u; ++k) {
            const float dl = probs_[k] - (k == actual_byte ? 1.0f : 0.0f);
            bout_[k] -= 0.05f * dl;
        }
    }

    forward_step(actual_byte, static_cast<std::uint16_t>(tok_id & kTokMask));
    ++chunk_pos_;
    ++chunk_filled_;
    ++step_;
}

}  // namespace fzip::fzac_v3
