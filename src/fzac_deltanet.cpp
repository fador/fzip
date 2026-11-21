// fzip — Online Byte-Level Gated DeltaNet (Fast-Weight Associative Matrix-Memory RNN)
// Bit-exact AVX2 + scalar fallback for FZAC v3.
#include "fzac_deltanet.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#define FZAC_DN_X64 1
#else
#define FZAC_DN_X64 0
#endif

namespace fzip::fzac_v3 {

namespace {

auto detect_avx2_dn() -> bool {
#if FZAC_DN_X64
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

#if FZAC_DN_X64
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

// Exact 8-lane dot product matching AVX2 & scalar tree reduction order
inline auto dot_exact(const float* a, const float* b, std::size_t n, bool use_avx2) -> float {
#if FZAC_DN_X64
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

// out[r] = sum_c M[r,c] * x[c] (if add==false) or out[r] += ... (if add==true)
void matvec_exact(float* out, const float* M, const float* x,
                  std::size_t rows, std::size_t cols, bool add, bool use_avx2) {
#if FZAC_DN_X64
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

// Combined gradient accumulation: gW += dy * x^T  AND  dx += W^T * dy
void bptt_linear_step(float* gW, const float* W, const float* dy,
                      const float* x, float* dx,
                      std::size_t rows, std::size_t cols, bool use_avx2) {
#if FZAC_DN_X64
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

// Delta Rule forward per head:
//   v_pred = S_prev * k_prev
//   e_err  = v - v_pred
//   S_cur  = alpha * S_prev + beta * e_err * k_prev^T
//   y_out  = S_cur * q
void delta_head_forward(float* S, const float* k_prev, const float* v, const float* q,
                        float alpha, float beta, float* e_err_out, float* y_out,
                        std::size_t dh, bool use_avx2) {
#if FZAC_DN_X64
    if (use_avx2) {
        const __m256 valpha = _mm256_set1_ps(alpha);
        for (std::size_t r = 0; r < dh; ++r) {
            float* srow = S + r * dh;
            __m256 acc_vp = _mm256_setzero_ps();
            for (std::size_t c = 0; c < dh; c += 8u) {
                acc_vp = _mm256_add_ps(acc_vp, _mm256_mul_ps(_mm256_loadu_ps(srow + c),
                                                             _mm256_loadu_ps(k_prev + c)));
            }
            const float err = v[r] - hsum256_exact(acc_vp);
            e_err_out[r] = err;
            const __m256 vbe = _mm256_set1_ps(beta * err);

            __m256 acc_y = _mm256_setzero_ps();
            for (std::size_t c = 0; c < dh; c += 8u) {
                const __m256 vs_old = _mm256_loadu_ps(srow + c);
                const __m256 vk = _mm256_loadu_ps(k_prev + c);
                const __m256 vs_new = _mm256_add_ps(_mm256_mul_ps(valpha, vs_old),
                                                    _mm256_mul_ps(vbe, vk));
                _mm256_storeu_ps(srow + c, vs_new);
                acc_y = _mm256_add_ps(acc_y, _mm256_mul_ps(vs_new, _mm256_loadu_ps(q + c)));
            }
            y_out[r] = hsum256_exact(acc_y);
        }
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t r = 0; r < dh; ++r) {
        float* srow = S + r * dh;
        float acc_vp[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (std::size_t c = 0; c < dh; c += 8u) {
            for (std::size_t j = 0; j < 8u; ++j) {
                acc_vp[j] = acc_vp[j] + (srow[c + j] * k_prev[c + j]);
            }
        }
        const float vp = ((acc_vp[0] + acc_vp[4]) + (acc_vp[1] + acc_vp[5])) +
                         ((acc_vp[2] + acc_vp[6]) + (acc_vp[3] + acc_vp[7]));
        const float err = v[r] - vp;
        e_err_out[r] = err;
        const float be = beta * err;

        float acc_y[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (std::size_t c = 0; c < dh; c += 8u) {
            for (std::size_t j = 0; j < 8u; ++j) {
                const float s_new = (alpha * srow[c + j]) + (be * k_prev[c + j]);
                srow[c + j] = s_new;
                acc_y[j] = acc_y[j] + (s_new * q[c + j]);
            }
        }
        y_out[r] = ((acc_y[0] + acc_y[4]) + (acc_y[1] + acc_y[5])) +
                   ((acc_y[2] + acc_y[6]) + (acc_y[3] + acc_y[7]));
    }
}

// Delta Rule backward per head (fused in single AVX2 pass!):
//   Inputs: dS [dh*dh] (in/out), S_after [dh*dh], S_prev [dh*dh] (nullptr if t==0),
//           dy [dh], q [dh], k_prev [dh], e_err [dh], alpha, beta
//   Outputs: dq [dh], dv [dh], dk_prev [dh], d_alpha_out, d_beta_out, updated dS
void delta_head_backward(float* dS, const float* S_after, const float* S_prev,
                         const float* dy, const float* q, const float* k_prev,
                         const float* e_err, float alpha, float beta,
                         float* dq_out, float* dv_out, float* dk_prev_out,
                         float& d_alpha_out, float& d_beta_out,
                         std::size_t dh, bool use_avx2) {
    // Zero outputs
    std::memset(dq_out, 0, dh * sizeof(float));
    std::memset(dk_prev_out, 0, dh * sizeof(float));
    float u[64];

#if FZAC_DN_X64
    if (use_avx2) {
        const __m256 valpha = _mm256_set1_ps(alpha);
        const __m256 vbeta = _mm256_set1_ps(beta);
        const __m256 vneg_beta = _mm256_set1_ps(-beta);
        const __m256 vclamp_lo = _mm256_set1_ps(-2.0f);
        const __m256 vclamp_hi = _mm256_set1_ps(2.0f);

        __m256 v_dalpha = _mm256_setzero_ps();
        float d_beta_acc = 0.0f;

        for (std::size_t r = 0; r < dh; ++r) {
            float* ds_row = dS + r * dh;
            const float* sa_row = S_after + r * dh;
            const float* sp_row = S_prev ? (S_prev + r * dh) : nullptr;
            const __m256 vdy = _mm256_set1_ps(dy[r]);
            const __m256 vbe = _mm256_set1_ps(beta * e_err[r]);

            __m256 acc_u = _mm256_setzero_ps();
            for (std::size_t c = 0; c < dh; c += 8u) {
                // dq += S_after^T * dy
                const __m256 vsa = _mm256_loadu_ps(sa_row + c);
                _mm256_storeu_ps(dq_out + c, _mm256_add_ps(_mm256_loadu_ps(dq_out + c), _mm256_mul_ps(vsa, vdy)));

                // dS_tot = dS + dy * q^T
                const __m256 vq = _mm256_loadu_ps(q + c);
                const __m256 vds = _mm256_add_ps(_mm256_loadu_ps(ds_row + c), _mm256_mul_ps(vdy, vq));
                _mm256_storeu_ps(ds_row + c, vds);

                // u[r] = sum_c dS_tot[r,c] * k_prev[c]
                acc_u = _mm256_add_ps(acc_u, _mm256_mul_ps(vds, _mm256_loadu_ps(k_prev + c)));

                // d_alpha += sum_{r,c} dS_tot[r,c] * S_prev[r,c]
                if (sp_row) {
                    v_dalpha = _mm256_add_ps(v_dalpha, _mm256_mul_ps(vds, _mm256_loadu_ps(sp_row + c)));
                }

                // dk_prev += beta * (dS_tot^T * e_err)
                _mm256_storeu_ps(dk_prev_out + c, _mm256_add_ps(_mm256_loadu_ps(dk_prev_out + c), _mm256_mul_ps(vds, vbe)));
            }
            const float ur = hsum256_exact(acc_u);
            u[r] = ur;
            dv_out[r] = beta * ur;
            d_beta_acc = d_beta_acc + (ur * e_err[r]);
        }
        d_alpha_out = hsum256_exact(v_dalpha);
        d_beta_out = d_beta_acc;

        // Second pass: dk_prev -= beta * (S_prev^T * u)  AND  dS = clamp(alpha * dS_tot - beta * u * k_prev^T)
        for (std::size_t r = 0; r < dh; ++r) {
            float* ds_row = dS + r * dh;
            const float* sp_row = S_prev ? (S_prev + r * dh) : nullptr;
            const __m256 vbu = _mm256_mul_ps(vneg_beta, _mm256_set1_ps(u[r]));
            for (std::size_t c = 0; c < dh; c += 8u) {
                if (sp_row) {
                    _mm256_storeu_ps(dk_prev_out + c,
                        _mm256_add_ps(_mm256_loadu_ps(dk_prev_out + c), _mm256_mul_ps(_mm256_loadu_ps(sp_row + c), vbu)));
                }
                const __m256 vds = _mm256_loadu_ps(ds_row + c);
                __m256 vnew = _mm256_add_ps(_mm256_mul_ps(valpha, vds),
                                            _mm256_mul_ps(vbu, _mm256_loadu_ps(k_prev + c)));
                vnew = _mm256_max_ps(vclamp_lo, _mm256_min_ps(vclamp_hi, vnew));
                _mm256_storeu_ps(ds_row + c, vnew);
            }
        }
        (void)vbeta;
        return;
    }
#else
    (void)use_avx2;
#endif
    float da_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float d_beta_acc = 0.0f;
    for (std::size_t r = 0; r < dh; ++r) {
        float* ds_row = dS + r * dh;
        const float* sa_row = S_after + r * dh;
        const float* sp_row = S_prev ? (S_prev + r * dh) : nullptr;
        const float dyr = dy[r];
        const float ber = beta * e_err[r];
        float acc_u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (std::size_t c = 0; c < dh; c += 8u) {
            for (std::size_t j = 0; j < 8u; ++j) {
                const std::size_t cj = c + j;
                dq_out[cj] = dq_out[cj] + (sa_row[cj] * dyr);
                const float vds = ds_row[cj] + (dyr * q[cj]);
                ds_row[cj] = vds;
                acc_u[j] = acc_u[j] + (vds * k_prev[cj]);
                if (sp_row) da_acc[j] = da_acc[j] + (vds * sp_row[cj]);
                dk_prev_out[cj] = dk_prev_out[cj] + (vds * ber);
            }
        }
        const float ur = ((acc_u[0] + acc_u[4]) + (acc_u[1] + acc_u[5])) +
                         ((acc_u[2] + acc_u[6]) + (acc_u[3] + acc_u[7]));
        u[r] = ur;
        dv_out[r] = beta * ur;
        d_beta_acc = d_beta_acc + (ur * e_err[r]);
    }
    d_alpha_out = ((da_acc[0] + da_acc[4]) + (da_acc[1] + da_acc[5])) +
                  ((da_acc[2] + da_acc[6]) + (da_acc[3] + da_acc[7]));
    d_beta_out = d_beta_acc;

    for (std::size_t r = 0; r < dh; ++r) {
        float* ds_row = dS + r * dh;
        const float* sp_row = S_prev ? (S_prev + r * dh) : nullptr;
        const float neg_bu = (-beta) * u[r];
        for (std::size_t c = 0; c < dh; ++c) {
            if (sp_row) dk_prev_out[c] = dk_prev_out[c] + (sp_row[c] * neg_bu);
            ds_row[c] = std::clamp((alpha * ds_row[c]) + (neg_bu * k_prev[c]), -2.0f, 2.0f);
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
#if FZAC_DN_X64
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
#if FZAC_DN_X64
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
#if FZAC_DN_X64
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
#if FZAC_DN_X64
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

void softmax_256(const float* logits, float* probs, bool use_avx2) {
    float max_l = logits[0];
    for (std::size_t k = 1u; k < 256u; ++k) {
        if (logits[k] > max_l) max_l = logits[k];
    }
    float sum_e = 0.0f;
#if FZAC_DN_X64
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
                              const float* h_concat, const float* probs,
                              std::uint8_t actual_byte, float* dh_out,
                              float lr, std::size_t total_H, bool use_avx2) {
    constexpr float kB1 = 0.50f, k1mB1 = 0.50f;
    constexpr float kB2 = 0.99f, k1mB2 = 0.01f;
    constexpr float kEps = 1e-4f;
    std::memset(dh_out, 0, total_H * sizeof(float));

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

#if FZAC_DN_X64
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
#if FZAC_DN_X64
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
#if FZAC_DN_X64
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

struct XorShift32Dn {
    std::uint32_t s = 0x9E3779B9u;
    auto next_f() -> float {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return static_cast<float>(static_cast<std::int32_t>(s)) * (1.0f / 2147483648.0f);
    }
};

}  // namespace

DeltaNetModel::DeltaNetModel(std::size_t num_heads, std::size_t d_head,
                             std::size_t d_in, std::size_t horizon)
    : num_heads_(std::clamp<std::size_t>(num_heads, 1u, 8u)),
      dh_(std::clamp<std::size_t>((d_head + 7u) & ~std::size_t{7}, 8u, 64u)),
      D_(num_heads_ * dh_),
      d_in_(std::clamp<std::size_t>((d_in + 7u) & ~std::size_t{7}, 16u, 128u)),
      total_H_(D_ + d_in_),
      horizon_(std::max<std::size_t>(8u, horizon)),
      stride_(horizon_),
      avx2_(detect_avx2_dn()) {
    XorShift32Dn rng;
    const float r_emb = 0.25f;
    const float r_proj = 1.0f / std::sqrt(static_cast<float>(d_in_));

    e_byte_.resize(256u * d_in_);
    m_ebyte_.assign(256u * d_in_, 0.0f);
    v_ebyte_.assign(256u * d_in_, 1e-3f);
    for (float& x : e_byte_) x = rng.next_f() * r_emb;

    e_tok_.assign(kTokBuckets * d_in_, 0.0f);
    v_etok_.assign(kTokBuckets * d_in_, 1e-3f);

    // 4-tap depthwise conv initialized near identity on current byte + decay on past 3 bytes
    conv_w_.assign(kConvK * d_in_, 0.0f);
    for (std::size_t i = 0; i < d_in_; ++i) {
        conv_w_[0u * d_in_ + i] = 0.85f + 0.10f * rng.next_f();
        conv_w_[1u * d_in_ + i] = 0.35f + 0.10f * rng.next_f();
        conv_w_[2u * d_in_ + i] = 0.15f + 0.08f * rng.next_f();
        conv_w_[3u * d_in_ + i] = 0.05f + 0.05f * rng.next_f();
    }
    m_convw_.assign(kConvK * d_in_, 0.0f);
    v_convw_.assign(kConvK * d_in_, 1e-3f);
    g_convw_.assign(kConvK * d_in_, 0.0f);
    conv_b_.assign(d_in_, 0.0f);
    m_convb_.assign(d_in_, 0.0f);
    v_convb_.assign(d_in_, 1e-3f);
    g_convb_.assign(d_in_, 0.0f);
    e_ring_.assign(kConvK * d_in_, 0.0f);

    gx_.assign(d_in_, 1.0f); m_gx_.assign(d_in_, 0.0f); v_gx_.assign(d_in_, 1e-3f); g_gx_.assign(d_in_, 0.0f);
    bx_.assign(d_in_, 0.0f); m_bx_.assign(d_in_, 0.0f); v_bx_.assign(d_in_, 1e-3f); g_bx_.assign(d_in_, 0.0f);
    gy_.assign(D_, 1.0f);    m_gy_.assign(D_, 0.0f);    v_gy_.assign(D_, 1e-3f);    g_gy_.assign(D_, 0.0f);
    by_.assign(D_, 0.0f);    m_by_.assign(D_, 0.0f);    v_by_.assign(D_, 1e-3f);    g_by_.assign(D_, 0.0f);

    const std::size_t proj_sz = D_ * d_in_;
    wq_.resize(proj_sz); wk_.resize(proj_sz); wv_.resize(proj_sz);
    for (std::size_t i = 0; i < proj_sz; ++i) {
        wq_[i] = rng.next_f() * r_proj;
        wk_[i] = rng.next_f() * r_proj;
        wv_[i] = rng.next_f() * r_proj;
    }
    // Seed W_Q and W_K close to each other initially so key==query retrieval works from byte 1!
    for (std::size_t i = 0; i < proj_sz; ++i) {
        wq_[i] = 0.75f * wk_[i] + 0.25f * wq_[i];
    }
    m_wq_.assign(proj_sz, 0.0f); v_wq_.assign(proj_sz, 1e-3f); g_wq_.assign(proj_sz, 0.0f);
    m_wk_.assign(proj_sz, 0.0f); v_wk_.assign(proj_sz, 1e-3f); g_wk_.assign(proj_sz, 0.0f);
    m_wv_.assign(proj_sz, 0.0f); v_wv_.assign(proj_sz, 1e-3f); g_wv_.assign(proj_sz, 0.0f);

    wa_.assign(num_heads_ * d_in_, 0.0f);
    wb_.assign(num_heads_ * d_in_, 0.0f);
    for (std::size_t i = 0; i < num_heads_ * d_in_; ++i) {
        wa_[i] = rng.next_f() * 0.1f;
        wb_[i] = rng.next_f() * 0.1f;
    }
    m_wa_.assign(num_heads_ * d_in_, 0.0f); v_wa_.assign(num_heads_ * d_in_, 1e-3f); g_wa_.assign(num_heads_ * d_in_, 0.0f);
    m_wb_.assign(num_heads_ * d_in_, 0.0f); v_wb_.assign(num_heads_ * d_in_, 1e-3f); g_wb_.assign(num_heads_ * d_in_, 0.0f);

    // Multi-timescale decay horizons across heads: fast local -> medium -> long-range!
    ba_.resize(num_heads_);
    bb_.assign(num_heads_, -0.5f);  // initial write strength ~0.38
    for (std::size_t h = 0; h < num_heads_; ++h) {
        ba_[h] = 1.0f + 1.5f * static_cast<float>(h);  // 1.0, 2.5, 4.0, 5.5
    }
    m_ba_.assign(num_heads_, 0.0f); v_ba_.assign(num_heads_, 1e-3f); g_ba_.assign(num_heads_, 0.0f);
    m_bb_.assign(num_heads_, 0.0f); v_bb_.assign(num_heads_, 1e-3f); g_bb_.assign(num_heads_, 0.0f);

    S_cur_.assign(num_heads_ * dh_ * dh_, 0.0f);
    k_prev_.assign(D_, 0.0f);

    wout_.assign(256u * total_H_, 0.0f);
    m_wout_.assign(256u * total_H_, 0.0f);
    v_wout_.assign(256u * total_H_, 1e-3f);
    bout_.assign(256u, 0.0f);
    m_bout_.assign(256u, 0.0f);
    v_bout_.assign(256u, 1e-3f);
    h_concat_.assign(total_H_, 0.0f);
    probs_.fill(1.0f / 256.0f);

    // Ring buffer of size horizon_ + 1 so S_{t-1} and k_{t-1} are always preserved!
    hist_.resize(horizon_ + 1u);
    for (auto& sc : hist_) {
        sc.conv_e.assign(kConvK * d_in_, 0.0f);
        sc.conv_u.assign(d_in_, 0.0f);
        sc.x_un.assign(d_in_, 0.0f);
        sc.x_hat.assign(d_in_, 0.0f);
        sc.q.assign(D_, 0.0f);
        sc.k.assign(D_, 0.0f);
        sc.inv_nq.assign(num_heads_, 1.0f);
        sc.inv_nk.assign(num_heads_, 1.0f);
        sc.v.assign(D_, 0.0f);
        sc.e_err.assign(D_, 0.0f);
        sc.alpha.assign(num_heads_, 0.9f);
        sc.beta.assign(num_heads_, 0.3f);
        sc.S_after.assign(num_heads_ * dh_ * dh_, 0.0f);
        sc.y_un.assign(D_, 0.0f);
        sc.dy_ext.assign(D_, 0.0f);
        sc.dx_ext.assign(d_in_, 0.0f);
    }

    dh_all_.assign(total_H_, 0.0f);
    dS_.assign(num_heads_ * dh_ * dh_, 0.0f);
    dk_next_.assign(D_, 0.0f);
    dq_raw_.assign(D_, 0.0f);
    dk_raw_.assign(D_, 0.0f);
    dv_.assign(D_, 0.0f);
    dx_hat_.assign(d_in_, 0.0f);
    du_conv_.assign(d_in_, 0.0f);
    de_shift_.assign(kConvK * d_in_, 0.0f);

    build_tree();
}

void DeltaNetModel::build_tree() {
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

auto DeltaNetModel::predict_bit(std::uint32_t c0) const -> std::uint32_t {
    const std::uint32_t tot = tree_[c0];
    const std::uint32_t p1 = tree_[c0 * 2u + 1u];
    const std::uint64_t p = (static_cast<std::uint64_t>(p1) * 4096u + (tot >> 1)) / tot;
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(p, 1u, 4095u));
}

void DeltaNetModel::forward_step(std::uint8_t in_byte, std::uint16_t tok_id) {
    auto& sc = hist_[hist_pos_];
    sc.in_byte = in_byte;
    sc.tok_id = tok_id;

    // 1. Shift 4-tap embedding history & form e_t = E_byte[in_byte] + E_tok[tok_id]
    for (std::size_t m = kConvK - 1u; m >= 1u; --m) {
        std::memcpy(e_ring_.data() + m * d_in_, e_ring_.data() + (m - 1u) * d_in_, d_in_ * sizeof(float));
    }
    const float* eb = e_byte_.data() + static_cast<std::size_t>(in_byte) * d_in_;
    const float* et = e_tok_.data() + static_cast<std::size_t>(tok_id) * d_in_;
    for (std::size_t i = 0; i < d_in_; ++i) {
        e_ring_[i] = eb[i] + et[i];
    }
    std::memcpy(sc.conv_e.data(), e_ring_.data(), kConvK * d_in_ * sizeof(float));

    // 2. Depthwise causal 4-tap conv -> SiLU -> RMSNorm -> x_hat
    for (std::size_t i = 0; i < d_in_; ++i) {
        float acc = conv_b_[i];
        for (std::size_t m = 0; m < kConvK; ++m) {
            acc = acc + (conv_w_[m * d_in_ + i] * e_ring_[m * d_in_ + i]);
        }
        sc.conv_u[i] = acc;
    }
    float x_silu[128];
    silu_forward(sc.conv_u.data(), x_silu, d_in_, avx2_);
    rmsnorm_forward(x_silu, gx_.data(), bx_.data(), sc.x_un.data(), sc.x_hat.data(), sc.inv_rms_x, d_in_, avx2_);

    // 3. Project Q, K, V [D] and gates alpha, beta [num_heads]
    matvec_exact(dq_raw_.data(), wq_.data(), sc.x_hat.data(), D_, d_in_, false, avx2_);
    matvec_exact(dk_raw_.data(), wk_.data(), sc.x_hat.data(), D_, d_in_, false, avx2_);
    matvec_exact(sc.v.data(),    wv_.data(), sc.x_hat.data(), D_, d_in_, false, avx2_);

    for (std::size_t h = 0; h < num_heads_; ++h) {
        const float* qr = dq_raw_.data() + h * dh_;
        const float* kr = dk_raw_.data() + h * dh_;
        const float inv_nq = 1.0f / std::sqrt(dot_exact(qr, qr, dh_, avx2_) + 1e-4f);
        const float inv_nk = 1.0f / std::sqrt(dot_exact(kr, kr, dh_, avx2_) + 1e-4f);
        sc.inv_nq[h] = inv_nq;
        sc.inv_nk[h] = inv_nk;
        float* qh = sc.q.data() + h * dh_;
        float* kh = sc.k.data() + h * dh_;
        for (std::size_t j = 0; j < dh_; ++j) {
            qh[j] = qr[j] * inv_nq;
            kh[j] = kr[j] * inv_nk;
        }
        const float la = ba_[h] + dot_exact(wa_.data() + h * d_in_, sc.x_hat.data(), d_in_, avx2_);
        const float lb = bb_[h] + dot_exact(wb_.data() + h * d_in_, sc.x_hat.data(), d_in_, avx2_);
        sc.alpha[h] = fast_sigmoid_scalar(la);
        sc.beta[h]  = fast_sigmoid_scalar(lb);
    }

    // 4. Associative Delta Rule state update + memory readout per head!
    float y_raw[256];
    for (std::size_t h = 0; h < num_heads_; ++h) {
        float* Sh = S_cur_.data() + h * dh_ * dh_;
        delta_head_forward(Sh, k_prev_.data() + h * dh_, sc.v.data() + h * dh_,
                           sc.q.data() + h * dh_, sc.alpha[h], sc.beta[h],
                           sc.e_err.data() + h * dh_, y_raw + h * dh_, dh_, avx2_);
    }
    std::memcpy(sc.S_after.data(), S_cur_.data(), num_heads_ * dh_ * dh_ * sizeof(float));
    std::memcpy(k_prev_.data(), sc.k.data(), D_ * sizeof(float));

    // 5. Output RMSNorm on y_raw -> concatenate [y_norm, x_hat] -> 256-way softmax
    rmsnorm_forward(y_raw, gy_.data(), by_.data(), sc.y_un.data(), h_concat_.data(), sc.inv_rms_y, D_, avx2_);
    std::memcpy(h_concat_.data() + D_, sc.x_hat.data(), d_in_ * sizeof(float));

    std::array<float, 256> logits{};
    std::memcpy(logits.data(), bout_.data(), 256u * sizeof(float));
    matvec_exact(logits.data(), wout_.data(), h_concat_.data(), 256u, total_H_, true, avx2_);
    softmax_256(logits.data(), probs_.data(), avx2_);
    build_tree();
}

void DeltaNetModel::run_bptt() {
    const std::size_t ring = hist_.size();
    const std::size_t steps = std::min(hist_filled_, horizon_);
    const float decay = 1.0f / std::sqrt(1.0f + static_cast<float>(step_) * (1.0f / 262144.0f));
    const float lr_emb  = ((step_ < 16384u) ? 0.040f : 0.022f) * decay;
    const float lr_tok  = ((step_ < 16384u) ? 0.048f : 0.026f) * decay;
    const float lr_proj = ((step_ < 16384u) ? 0.024f : 0.013f) * decay;

    std::fill(dS_.begin(), dS_.end(), 0.0f);
    std::fill(dk_next_.begin(), dk_next_.end(), 0.0f);
    std::fill(de_shift_.begin(), de_shift_.end(), 0.0f);

    float zero_k[64] = {};
    std::size_t idx = (hist_pos_ + ring - 1u) % ring;
    for (std::size_t s = 0; s < steps; ++s) {
        auto& sc = hist_[idx];
        const bool has_prev = (step_ > steps - s);
        const std::size_t prev_idx = (idx + ring - 1u) % ring;
        const float* S_prev_all = has_prev ? hist_[prev_idx].S_after.data() : nullptr;
        const float* k_prev_all = has_prev ? hist_[prev_idx].k.data() : nullptr;

        // Backprop readout dy_ext through output RMSNorm
        float dy_raw[256];
        std::memcpy(dy_raw, sc.dy_ext.data(), D_ * sizeof(float));
        rmsnorm_backward_inplace(dy_raw, sc.y_un.data(), gy_.data(), sc.inv_rms_y,
                                 g_gy_.data(), g_by_.data(), D_, avx2_);

        float dk_for_prev[256];
        float da_alpha[8];
        float da_beta[8];

        // Backprop per head through Delta Rule & readout
        for (std::size_t h = 0; h < num_heads_; ++h) {
            const float* sp = S_prev_all ? (S_prev_all + h * dh_ * dh_) : nullptr;
            const float* kp = k_prev_all ? (k_prev_all + h * dh_) : zero_k;
            float dq_h[64];
            float d_al = 0.0f, d_be = 0.0f;
            delta_head_backward(
                dS_.data() + h * dh_ * dh_,
                sc.S_after.data() + h * dh_ * dh_,
                sp,
                dy_raw + h * dh_,
                sc.q.data() + h * dh_,
                kp,
                sc.e_err.data() + h * dh_,
                sc.alpha[h], sc.beta[h],
                dq_h,
                dv_.data() + h * dh_,
                dk_for_prev + h * dh_,
                d_al, d_be,
                dh_, avx2_);

            // Gate sigmoid derivatives
            da_alpha[h] = std::clamp(d_al * sc.alpha[h] * (1.0f - sc.alpha[h]), -1.5f, 1.5f);
            da_beta[h]  = std::clamp(d_be * sc.beta[h]  * (1.0f - sc.beta[h]),  -1.5f, 1.5f);

            // L2-norm backward for q_t and k_t (using dk_next_ received from step t+1)
            const float* qh = sc.q.data() + h * dh_;
            const float* kh = sc.k.data() + h * dh_;
            const float* dkh = dk_next_.data() + h * dh_;
            const float dot_dq = dot_exact(dq_h, qh, dh_, avx2_);
            const float dot_dk = dot_exact(dkh, kh, dh_, avx2_);
            const float inq = sc.inv_nq[h];
            const float ink = sc.inv_nk[h];
            float* dqr = dq_raw_.data() + h * dh_;
            float* dkr = dk_raw_.data() + h * dh_;
            for (std::size_t j = 0; j < dh_; ++j) {
                dqr[j] = std::clamp(inq * (dq_h[j] - qh[j] * dot_dq), -1.5f, 1.5f);
                dkr[j] = std::clamp(ink * (dkh[j]  - kh[j] * dot_dk), -1.5f, 1.5f);
            }
        }
        std::memcpy(dk_next_.data(), dk_for_prev, D_ * sizeof(float));

        // Linear projections backward -> dx_hat
        std::memcpy(dx_hat_.data(), sc.dx_ext.data(), d_in_ * sizeof(float));
        bptt_linear_step(g_wq_.data(), wq_.data(), dq_raw_.data(), sc.x_hat.data(), dx_hat_.data(), D_, d_in_, avx2_);
        bptt_linear_step(g_wk_.data(), wk_.data(), dk_raw_.data(), sc.x_hat.data(), dx_hat_.data(), D_, d_in_, avx2_);
        bptt_linear_step(g_wv_.data(), wv_.data(), dv_.data(),     sc.x_hat.data(), dx_hat_.data(), D_, d_in_, avx2_);

        for (std::size_t h = 0; h < num_heads_; ++h) {
            const float da = da_alpha[h];
            const float db = da_beta[h];
            g_ba_[h] = g_ba_[h] + da;
            g_bb_[h] = g_bb_[h] + db;
            float* gwa = g_wa_.data() + h * d_in_;
            float* gwb = g_wb_.data() + h * d_in_;
            const float* wa = wa_.data() + h * d_in_;
            const float* wb = wb_.data() + h * d_in_;
            for (std::size_t i = 0; i < d_in_; ++i) {
                gwa[i] = gwa[i] + (da * sc.x_hat[i]);
                gwb[i] = gwb[i] + (db * sc.x_hat[i]);
                dx_hat_[i] = dx_hat_[i] + (da * wa[i]) + (db * wb[i]);
            }
        }

        // Backprop through input RMSNorm & SiLU -> du_conv_
        std::memcpy(du_conv_.data(), dx_hat_.data(), d_in_ * sizeof(float));
        rmsnorm_backward_inplace(du_conv_.data(), sc.x_un.data(), gx_.data(), sc.inv_rms_x,
                                 g_gx_.data(), g_bx_.data(), d_in_, avx2_);
        silu_backward_inplace(du_conv_.data(), sc.conv_u.data(), d_in_, avx2_);

        // Backprop through 4-tap depthwise conv into de_shift_
        for (std::size_t i = 0; i < d_in_; ++i) {
            const float du = std::clamp(du_conv_[i], -1.5f, 1.5f);
            g_convb_[i] = g_convb_[i] + du;
            for (std::size_t m = 0; m < kConvK; ++m) {
                g_convw_[m * d_in_ + i] = g_convw_[m * d_in_ + i] + (du * sc.conv_e[m * d_in_ + i]);
                de_shift_[m * d_in_ + i] = de_shift_[m * d_in_ + i] + (du * conv_w_[m * d_in_ + i]);
            }
        }

        // Apply accumulated embedding gradient de_shift_[0] to E_byte & E_tok
        const std::size_t toff = static_cast<std::size_t>(sc.tok_id) * d_in_;
        rmsprop_update_sparse(e_tok_.data() + toff, v_etok_.data() + toff, de_shift_.data(), lr_tok, d_in_, avx2_);

        const std::size_t eoff = static_cast<std::size_t>(sc.in_byte) * d_in_;
        adam_update_grad(e_byte_.data() + eoff, m_ebyte_.data() + eoff, v_ebyte_.data() + eoff,
                         de_shift_.data(), lr_emb, d_in_, avx2_, false);

        // Shift de_shift_ backwards by 1 tap
        for (std::size_t m = 0; m + 1u < kConvK; ++m) {
            std::memcpy(de_shift_.data() + m * d_in_, de_shift_.data() + (m + 1u) * d_in_, d_in_ * sizeof(float));
        }
        std::memset(de_shift_.data() + (kConvK - 1u) * d_in_, 0, d_in_ * sizeof(float));

        idx = prev_idx;
    }

    // Apply Adam updates to all dense slow weights
    adam_update_grad(wq_.data(), m_wq_.data(), v_wq_.data(), g_wq_.data(), lr_proj, D_ * d_in_, avx2_);
    adam_update_grad(wk_.data(), m_wk_.data(), v_wk_.data(), g_wk_.data(), lr_proj, D_ * d_in_, avx2_);
    adam_update_grad(wv_.data(), m_wv_.data(), v_wv_.data(), g_wv_.data(), lr_proj, D_ * d_in_, avx2_);
    adam_update_grad(wa_.data(), m_wa_.data(), v_wa_.data(), g_wa_.data(), lr_proj, num_heads_ * d_in_, avx2_);
    adam_update_grad(wb_.data(), m_wb_.data(), v_wb_.data(), g_wb_.data(), lr_proj, num_heads_ * d_in_, avx2_);
    adam_update_grad(ba_.data(), m_ba_.data(), v_ba_.data(), g_ba_.data(), lr_proj, num_heads_, false);
    adam_update_grad(bb_.data(), m_bb_.data(), v_bb_.data(), g_bb_.data(), lr_proj, num_heads_, false);
    adam_update_grad(conv_w_.data(), m_convw_.data(), v_convw_.data(), g_convw_.data(), lr_proj, kConvK * d_in_, avx2_);
    adam_update_grad(conv_b_.data(), m_convb_.data(), v_convb_.data(), g_convb_.data(), lr_proj, d_in_, avx2_);
    adam_update_grad(gx_.data(), m_gx_.data(), v_gx_.data(), g_gx_.data(), lr_proj, d_in_, avx2_);
    adam_update_grad(bx_.data(), m_bx_.data(), v_bx_.data(), g_bx_.data(), lr_proj, d_in_, avx2_);
    adam_update_grad(gy_.data(), m_gy_.data(), v_gy_.data(), g_gy_.data(), lr_proj, D_, avx2_);
    adam_update_grad(by_.data(), m_by_.data(), v_by_.data(), g_by_.data(), lr_proj, D_, avx2_);
}

void DeltaNetModel::update_byte(std::uint8_t actual_byte, std::uint32_t tok_id) {
    const float decay = 1.0f / std::sqrt(1.0f + static_cast<float>(step_) * (1.0f / 262144.0f));
    const float lr_out = ((step_ < 16384u) ? 0.045f : 0.024f) * decay;
    const std::size_t ring = hist_.size();

    if (hist_filled_ > 0u) {
        const std::size_t prev_idx = (hist_pos_ + ring - 1u) % ring;
        auto& prev_sc = hist_[prev_idx];

        update_output_layer_adam(wout_.data(), m_wout_.data(), v_wout_.data(),
                                 bout_.data(), m_bout_.data(), v_bout_.data(),
                                 h_concat_.data(), probs_.data(), actual_byte,
                                 dh_all_.data(), lr_out, total_H_, avx2_);
        std::memcpy(prev_sc.dy_ext.data(), dh_all_.data(), D_ * sizeof(float));
        std::memcpy(prev_sc.dx_ext.data(), dh_all_.data() + D_, d_in_ * sizeof(float));

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
    hist_pos_ = (hist_pos_ + 1u) % ring;
    if (hist_filled_ < horizon_) ++hist_filled_;
    ++step_;
}

}  // namespace fzip::fzac_v3
