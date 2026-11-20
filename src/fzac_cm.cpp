// fzip — FZAC v3 State-of-the-Art Context Mixing (CM) engine.
// Combines:
//   - Dynamic Universal WRT (Word & Structural Token Transform) & Bijective x86/x64 BCJ+Jcc
//   - Exact Direct O0/O1/O2 with bit-position-indexed StateMaps
//   - Split Low/High 3-way 64-byte Cache-Line ContextMap (Orders 3..15 + Sparse + Word/XML + Record + Indirect + Exe)
//   - 4-Way Associative Long-Range Dual MatchModel with 1-byte Edit Recovery
//   - Variable-Order Byte-Level PPMd (up to Order-16) with Full Exclusion & SEE
//   - Online Byte-Level Coupled-Gate LSTM Neural Predictor (AVX2 bit-exact, levels 10..12)
//   - 2-Stage Multi-Context AVX2 Logistic Mixer (48 inputs -> 4 Stage-1 -> Meta-Mixer)
//   - 3-Stage Count-Adaptive Cascaded APM / SSE + 16-bit Precision Carryless Range Coder
#include "codec.hpp"
#include "fzac_cm.hpp"
#include "fzac_lstm.hpp"
#include "fzac_ppm.hpp"
#include "fzac_wrt.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#define FZAC_X64 1
#else
#define FZAC_X64 0
#endif

namespace fzip::fzac_v3 {

namespace {

// ---------------------------------------------------------------------------
// CPU Feature Detection (AVX2 runtime check for bit-exact SIMD acceleration)
// ---------------------------------------------------------------------------
auto detect_avx2() -> bool {
#if FZAC_X64
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

inline auto has_avx2() -> bool {
    static const bool kAvx2 = detect_avx2();
    return kAvx2;
}

// ---------------------------------------------------------------------------
// Fast Unaligned Loads, Stores, Varints, Prefetch & Hashing
// ---------------------------------------------------------------------------
inline auto load_u16(const std::uint8_t* p) -> std::uint16_t {
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

inline auto load_u32(const std::uint8_t* p) -> std::uint32_t {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline auto load_u64(const std::uint8_t* p) -> std::uint64_t {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

inline void store_u32(std::uint8_t* p, std::uint32_t v) {
    std::memcpy(p, &v, 4);
}

inline void prefetch_l1(const void* ptr) {
#if FZAC_X64
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#else
    (void)ptr;
#endif
}

inline auto mix64(std::uint64_t x) -> std::uint64_t {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

inline auto hash_combine(std::uint64_t h, std::uint64_t v) -> std::uint64_t {
    return (h * 0x9E3779B97F4A7C15ULL) ^ mix64(v + 0x517CC1B727220A95ULL);
}

inline void append_varint(std::vector<std::byte>& out, std::uint64_t v) {
    while (v >= 0x80u) {
        out.push_back(static_cast<std::byte>((v & 0x7Fu) | 0x80u));
        v >>= 7;
    }
    out.push_back(static_cast<std::byte>(v));
}

inline auto read_varint(const std::uint8_t* in, std::size_t size,
                        std::size_t& off, const char* err_msg) -> std::uint64_t {
    std::uint64_t val = 0;
    int shift = 0;
    for (;;) {
        if (off >= size || shift > 63) {
            throw std::runtime_error(err_msg);
        }
        const std::uint8_t b = in[off++];
        val |= static_cast<std::uint64_t>(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0u) return val;
        shift += 7;
    }
}

inline auto byte_class(std::uint8_t b) -> std::uint32_t {
    if (b <= 32u) return 0u;
    if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')) return 1u;
    if (b >= '0' && b <= '9') return 2u;
    return 3u;
}

// ---------------------------------------------------------------------------
// 16-Bit Precision Carryless Range Encoder & Decoder
// ---------------------------------------------------------------------------
constexpr std::uint32_t kTopValue = 1u << 24;

class RangeEncoder16 {
  public:
    void reserve(std::size_t bytes) { out_.reserve(bytes); }

    void encode_bit(std::uint32_t p16, int bit) {
        const std::uint32_t bound =
            static_cast<std::uint32_t>((static_cast<std::uint64_t>(range_) * p16) >> 16);
        if (bit != 0) {
            range_ = bound;
        } else {
            low_ += bound;
            range_ -= bound;
        }
        while (range_ < kTopValue) {
            range_ <<= 8;
            shift_low();
        }
    }

    void finish() {
        for (int i = 0; i < 5; ++i) {
            shift_low();
        }
    }

    auto buffer() -> std::vector<std::byte>& { return out_; }

  private:
    void emit_byte(std::uint8_t b) {
        if (!skipped_first_) {
            skipped_first_ = true;
            return;
        }
        out_.push_back(static_cast<std::byte>(b));
    }

    void shift_low() {
        const std::uint32_t low_hi = static_cast<std::uint32_t>(low_ >> 32);
        if (low_hi != 0u || low_ < 0xFF000000ULL) {
            std::uint8_t temp = cache_;
            do {
                emit_byte(static_cast<std::uint8_t>(temp + low_hi));
                temp = 0xFFu;
            } while (--cache_size_ != 0u);
            cache_ = static_cast<std::uint8_t>(static_cast<std::uint32_t>(low_) >> 24);
        }
        ++cache_size_;
        low_ = (low_ & 0x00FFFFFFULL) << 8;
    }

    std::vector<std::byte> out_;
    std::uint64_t low_ = 0;
    std::uint32_t range_ = 0xFFFFFFFFu;
    std::uint64_t cache_size_ = 1;
    std::uint8_t cache_ = 0;
    bool skipped_first_ = false;
};

class RangeDecoder16 {
  public:
    RangeDecoder16(const std::byte* begin, const std::byte* end)
        : pos_(reinterpret_cast<const std::uint8_t*>(begin)),
          end_(reinterpret_cast<const std::uint8_t*>(end)) {
        for (int i = 0; i < 4; ++i) {
            code_ = (code_ << 8) | next_byte();
        }
    }

    auto decode_bit(std::uint32_t p16) -> int {
        const std::uint32_t bound =
            static_cast<std::uint32_t>((static_cast<std::uint64_t>(range_) * p16) >> 16);
        int bit;
        if (code_ < bound) {
            range_ = bound;
            bit = 1;
        } else {
            code_ -= bound;
            range_ -= bound;
            bit = 0;
        }
        while (range_ < kTopValue) {
            range_ <<= 8;
            code_ = (code_ << 8) | next_byte();
        }
        return bit;
    }

  private:
    auto next_byte() -> std::uint32_t {
        return (pos_ < end_) ? static_cast<std::uint32_t>(*pos_++) : 0u;
    }

    const std::uint8_t* pos_;
    const std::uint8_t* end_;
    std::uint32_t range_ = 0xFFFFFFFFu;
    std::uint32_t code_ = 0;
};

// ---------------------------------------------------------------------------
// Probability Math Tables: Stretch, Squash12, Squash16, Update Step Sizes
// ---------------------------------------------------------------------------
struct ProbTables {
    std::array<std::int16_t, 4096> stretch{};
    std::array<std::uint16_t, 4096> squash12{};
    std::array<std::uint16_t, 4096> squash16{};
    std::array<std::int32_t, 1024> dt{};
    std::array<std::int16_t, 256> run_logit{};

    ProbTables() {
        for (int p = 0; p < 4096; ++p) {
            const double prob = (static_cast<double>(p) + 0.5) / 4096.0;
            const double s = std::log(prob / (1.0 - prob)) * 256.0;
            stretch[static_cast<std::size_t>(p)] =
                static_cast<std::int16_t>(std::clamp(std::llround(s), -2047LL, 2047LL));
        }
        for (int i = 0; i < 4096; ++i) {
            const double x = static_cast<double>(i - 2048) / 256.0;
            const double sig = 1.0 / (1.0 + std::exp(-x));
            squash12[static_cast<std::size_t>(i)] =
                static_cast<std::uint16_t>(std::clamp(std::llround(sig * 4096.0), 1LL, 4095LL));
            squash16[static_cast<std::size_t>(i)] =
                static_cast<std::uint16_t>(std::clamp(std::llround(sig * 65536.0), 1LL, 65535LL));
        }
        for (int n = 0; n < 1024; ++n) {
            dt[static_cast<std::size_t>(n)] = std::max(2, 2048 / (n + 2));
        }
        run_logit[0] = 0;
        for (int r = 1; r < 256; ++r) {
            const double lo = std::log(static_cast<double>(r) * 2.0 + 1.0) * 256.0;
            run_logit[static_cast<std::size_t>(r)] =
                static_cast<std::int16_t>(std::clamp(std::llround(lo), 0LL, 2047LL));
        }
    }
};

inline auto prob_tables() -> const ProbTables& {
    static const ProbTables kTables;
    return kTables;
}

inline auto stretch_p12(std::uint32_t p12) -> std::int16_t {
    return prob_tables().stretch[p12 & 4095u];
}

inline auto squash_to_12(std::int32_t x) -> std::uint32_t {
    const std::int32_t c = std::clamp(x, -2047, 2047);
    return prob_tables().squash12[static_cast<std::size_t>(c + 2048)];
}

inline auto squash_to_16(std::int32_t x) -> std::uint32_t {
    const std::int32_t c = std::clamp(x, -2047, 2047);
    return prob_tables().squash16[static_cast<std::size_t>(c + 2048)];
}

// ---------------------------------------------------------------------------
// 256-State Bit-History State Machine (PAQ8-style)
// ---------------------------------------------------------------------------
struct StateTable {
    std::array<std::array<std::uint8_t, 2>, 256> nex{};
    std::array<std::uint32_t, 256> init_p20{};
    std::array<std::uint8_t, 256> priority{};

    StateTable() {
        struct Desc {
            int n0;
            int n1;
            int last;
        };
        std::vector<Desc> states;
        states.reserve(256);

        states.push_back({0, 0, -1});

        constexpr int kPure[] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            18, 20, 23, 27, 32, 39, 48, 60, 76, 98, 130, 180
        };
        for (int n : kPure) {
            states.push_back({n, 0, -1});
            states.push_back({0, n, -1});
        }

        constexpr int kRecPairs[][2] = {
            {1, 1}, {2, 1}, {1, 2}, {2, 2}, {3, 1}, {1, 3},
            {3, 2}, {2, 3}, {4, 1}, {1, 4}, {3, 3}, {4, 2}, {2, 4}
        };
        for (const auto& pr : kRecPairs) {
            states.push_back({pr[0], pr[1], 0});
            states.push_back({pr[0], pr[1], 1});
        }

        constexpr int kGrid[] = {1, 2, 3, 4, 5, 6, 7, 9, 12, 16, 22, 30, 42, 60, 88};
        for (int a : kGrid) {
            for (int b : kGrid) {
                if (states.size() >= 256u) break;
                bool exists = false;
                for (const auto& s : states) {
                    if (s.n0 == a && s.n1 == b) {
                        exists = true;
                        break;
                    }
                }
                if (!exists && (a + b <= 110)) {
                    states.push_back({a, b, -1});
                }
            }
        }
        int extra = 100;
        while (states.size() < 256u) {
            states.push_back({extra, extra / 2, -1});
            extra += 7;
        }

        auto find_nearest = [&](int t0, int t1, int bit) -> std::uint8_t {
            for (std::size_t i = 1; i < states.size(); ++i) {
                if (states[i].n0 == t0 && states[i].n1 == t1 && states[i].last == bit) {
                    return static_cast<std::uint8_t>(i);
                }
            }
            for (std::size_t i = 1; i < states.size(); ++i) {
                if (states[i].n0 == t0 && states[i].n1 == t1 && states[i].last == -1) {
                    return static_cast<std::uint8_t>(i);
                }
            }
            const double target_p = (static_cast<double>(t1) + 0.5) /
                                    (static_cast<double>(t0 + t1) + 1.0);
            const double target_logn = std::log(static_cast<double>(t0 + t1) + 1.0);
            std::size_t best_i = 1;
            double best_err = 1e30;
            for (std::size_t i = 1; i < states.size(); ++i) {
                if (states[i].last != -1 && states[i].last != bit) continue;
                const bool target_pure = (t0 == 0 || t1 == 0);
                const bool cand_pure = (states[i].n0 == 0 || states[i].n1 == 0);
                if (target_pure != cand_pure) continue;
                const double p = (static_cast<double>(states[i].n1) + 0.5) /
                                 (static_cast<double>(states[i].n0 + states[i].n1) + 1.0);
                const double logn = std::log(static_cast<double>(states[i].n0 + states[i].n1) + 1.0);
                const double err = std::abs(p - target_p) * 16.0 +
                                   std::abs(logn - target_logn);
                if (err < best_err) {
                    best_err = err;
                    best_i = i;
                }
            }
            return static_cast<std::uint8_t>(best_i);
        };

        for (std::size_t i = 0; i < 256u; ++i) {
            const auto& s = states[i];
            for (int bit = 0; bit <= 1; ++bit) {
                int n0 = s.n0;
                int n1 = s.n1;
                if (bit == 0) {
                    ++n0;
                    if (n1 > 2) n1 = 2 + ((n1 - 2) >> 1);
                    if (n0 > 180) {
                        n0 = (n0 + 1) >> 1;
                        n1 = (n1 + 1) >> 1;
                    }
                } else {
                    ++n1;
                    if (n0 > 2) n0 = 2 + ((n0 - 2) >> 1);
                    if (n1 > 180) {
                        n0 = (n0 + 1) >> 1;
                        n1 = (n1 + 1) >> 1;
                    }
                }
                nex[i][static_cast<std::size_t>(bit)] = find_nearest(n0, n1, bit);
            }

            double p0;
            if (s.n0 == 0 && s.n1 == 0) {
                p0 = 0.5;
            } else if (s.n0 == 0) {
                p0 = (static_cast<double>(s.n1) + 0.25) / (static_cast<double>(s.n1) + 0.5);
            } else if (s.n1 == 0) {
                p0 = 0.25 / (static_cast<double>(s.n0) + 0.5);
            } else {
                const double rec_bias = (s.last == 1) ? 0.15 : ((s.last == 0) ? -0.15 : 0.0);
                p0 = (static_cast<double>(s.n1) + 0.5 + rec_bias) /
                     (static_cast<double>(s.n0 + s.n1) + 1.0);
            }
            const double clamped = std::clamp(p0, 0.0005, 0.9995);
            init_p20[i] = static_cast<std::uint32_t>(std::llround(clamped * 1048575.0));
            priority[i] = static_cast<std::uint8_t>(std::min(s.n0 + s.n1, 255));
        }
    }
};

inline auto state_table() -> const StateTable& {
    static const StateTable kST;
    return kST;
}

// ---------------------------------------------------------------------------
// Fast 32-bit Adaptive StateMap & FlatProbMap
// ---------------------------------------------------------------------------
class StateMap {
  public:
    explicit StateMap(std::size_t num_groups = 1, std::uint32_t limit = 511u)
        : t_(num_groups * 256u), limit_(std::min(limit, 1023u)) {
        const auto& st = state_table();
        for (std::size_t g = 0; g < num_groups; ++g) {
            for (std::size_t s = 0; s < 256u; ++s) {
                t_[g * 256u + s] = (st.init_p20[s] << 12) | 4u;
            }
        }
    }

    inline auto predict(std::size_t idx) const -> std::uint32_t {
        return std::clamp<std::uint32_t>(t_[idx] >> 20, 1u, 4095u);
    }

    inline void update(std::size_t idx, int y) {
        const std::uint32_t entry = t_[idx];
        std::uint32_t n = entry & 1023u;
        std::int32_t p = static_cast<std::int32_t>(entry >> 12);
        const std::int32_t target = y ? 1048575 : 0;
        p += ((target - p) * prob_tables().dt[n]) >> 11;
        if (n < limit_) ++n;
        t_[idx] = (static_cast<std::uint32_t>(std::clamp(p, 1, 1048574)) << 12) | n;
    }

  private:
    std::vector<std::uint32_t> t_;
    std::uint32_t limit_;
};

class FlatProbMap {
  public:
    explicit FlatProbMap(std::size_t size, std::uint32_t limit = 255u)
        : t_(size, (524288u << 12) | 2u), limit_(std::min(limit, 1023u)) {}

    void set_prior(std::size_t idx, double p0, std::uint32_t init_n = 2u) {
        const std::uint32_t p20 = static_cast<std::uint32_t>(
            std::llround(std::clamp(p0, 0.0005, 0.9995) * 1048575.0));
        t_[idx] = (p20 << 12) | std::min(init_n, 1023u);
    }

    inline auto predict(std::size_t idx) const -> std::uint32_t {
        return std::clamp<std::uint32_t>(t_[idx] >> 20, 1u, 4095u);
    }

    inline void update(std::size_t idx, int y) {
        const std::uint32_t entry = t_[idx];
        std::uint32_t n = entry & 1023u;
        std::int32_t p = static_cast<std::int32_t>(entry >> 12);
        const std::int32_t target = y ? 1048575 : 0;
        p += ((target - p) * prob_tables().dt[n]) >> 11;
        if (n < limit_) ++n;
        t_[idx] = (static_cast<std::uint32_t>(std::clamp(p, 1, 1048574)) << 12) | n;
    }

  private:
    std::vector<std::uint32_t> t_;
    std::uint32_t limit_;
};

// ---------------------------------------------------------------------------
// 3-Way Set-Associative Cache-Line Bucketed ContextMap
// ---------------------------------------------------------------------------
struct alignas(64) CmBucket {
    struct Slot {
        std::uint16_t tag = 0;
        std::uint8_t run_byte = 0;
        std::uint8_t run_cnt = 0;
        std::uint8_t states[15]{};
        std::uint8_t pad_slot = 0;
    };
    Slot slots[3]{};
    std::uint8_t pad[4]{};
};
static_assert(sizeof(CmBucket) == 64);

class ContextMap {
  public:
    ContextMap(std::size_t table_bytes, std::size_t num_models)
        : num_models_(num_models),
          sm_(num_models, 511u) {
        std::size_t num_buckets = std::max<std::size_t>(1024u, table_bytes / sizeof(CmBucket));
        num_buckets = std::bit_floor(num_buckets);
        buckets_.resize(num_buckets);
        mask_ = num_buckets - 1u;

        hashes_.assign(num_models, 0u);
        byte_slots_.assign(num_models, nullptr);
        byte_tags_.assign(num_models, 0u);
        cur_slots_.assign(num_models, nullptr);
        run_bytes_.assign(num_models, 0u);
        run_cnts_.assign(num_models, 0u);
        run_ok_.assign(num_models, 0u);
        last_states_.assign(num_models, 0u);
    }

    void lookup_byte(const std::uint64_t* hashes) {
        const auto& st = state_table();
        constexpr std::size_t kPrefetchDist = 6u;
        for (std::size_t i = 0; i < std::min(kPrefetchDist, num_models_); ++i) {
            hashes_[i] = hashes[i];
            prefetch_l1(&buckets_[hashes[i] & mask_]);
        }
        for (std::size_t i = 0; i < num_models_; ++i) {
            if (i + kPrefetchDist < num_models_) {
                hashes_[i + kPrefetchDist] = hashes[i + kPrefetchDist];
                prefetch_l1(&buckets_[hashes[i + kPrefetchDist] & mask_]);
            }
            const std::uint64_t h = hashes_[i];
            const std::uint16_t tag = static_cast<std::uint16_t>((h >> 48) | 1u);
            auto* slot = find_or_alloc(buckets_[h & mask_], tag, st);
            byte_slots_[i] = slot;
            byte_tags_[i] = tag;
            cur_slots_[i] = slot;
            run_bytes_[i] = slot->run_byte;
            run_cnts_[i] = slot->run_cnt;
            run_ok_[i] = (slot->run_cnt > 0u) ? 1u : 0u;
        }
    }

    void lookup_low_nibble(std::uint32_t hi_nibble) {
        const std::uint64_t nmix = mix64(static_cast<std::uint64_t>(hi_nibble) + 0xA5A5A5A5ULL);
        const auto& st = state_table();
        constexpr std::size_t kPrefetchDist = 6u;
        for (std::size_t i = 0; i < std::min(kPrefetchDist, num_models_); ++i) {
            prefetch_l1(&buckets_[(hashes_[i] ^ nmix) & mask_]);
        }
        for (std::size_t i = 0; i < num_models_; ++i) {
            if (i + kPrefetchDist < num_models_) {
                prefetch_l1(&buckets_[(hashes_[i + kPrefetchDist] ^ nmix) & mask_]);
            }
            const std::uint64_t h = hashes_[i] ^ nmix;
            const std::uint16_t tag = static_cast<std::uint16_t>((h >> 48) | 1u);
            cur_slots_[i] = find_or_alloc(buckets_[h & mask_], tag, st);
        }
    }

    auto predict(int bpos, std::uint32_t sub, std::int16_t* out,
                 std::size_t& out_len, std::int16_t& best_run_out) -> std::uint32_t {
        const std::uint32_t sidx = sub - 1u;
        const auto& pt = prob_tables();
        std::uint32_t hit_cnt = 0;
        std::int32_t best_run_lo = 0;
        std::int32_t best_run_abs = 0;

        for (std::size_t i = 0; i < num_models_; ++i) {
            const std::uint8_t s = cur_slots_[i]->states[sidx];
            last_states_[i] = s;
            if (s != 0u) {
                ++hit_cnt;
                out[out_len++] = stretch_p12(sm_.predict(i * 256u + s));
            } else {
                out[out_len++] = 0;
            }
            if (run_ok_[i]) {
                const int eb = (run_bytes_[i] >> (7 - bpos)) & 1;
                const std::int32_t lo = pt.run_logit[run_cnts_[i]];
                if (lo > best_run_abs) {
                    best_run_abs = lo;
                    best_run_lo = eb ? lo : -lo;
                }
            }
        }
        best_run_out = static_cast<std::int16_t>(best_run_lo);
        return hit_cnt;
    }

    void update_bit(int bpos, std::uint32_t sub, int bit) {
        const std::uint32_t sidx = sub - 1u;
        const auto& st = state_table();
        for (std::size_t i = 0; i < num_models_; ++i) {
            const std::uint8_t s = last_states_[i];
            if (s != 0u) sm_.update(i * 256u + s, bit);
            cur_slots_[i]->states[sidx] = st.nex[s][static_cast<std::size_t>(bit)];
            if (run_ok_[i]) {
                const int eb = (run_bytes_[i] >> (7 - bpos)) & 1;
                if (bit != eb) run_ok_[i] = 0u;
            }
        }
    }

    void update_byte(std::uint8_t byte) {
        for (std::size_t i = 0; i < num_models_; ++i) {
            auto* slot = byte_slots_[i];
            if (slot != nullptr && slot->tag == byte_tags_[i]) {
                if (slot->run_cnt > 0u && slot->run_byte == byte) {
                    if (slot->run_cnt < 255u) ++slot->run_cnt;
                } else {
                    slot->run_byte = byte;
                    slot->run_cnt = 1u;
                }
            }
        }
    }

  private:
    static auto find_or_alloc(CmBucket& b, std::uint16_t tag,
                              const StateTable& st) -> CmBucket::Slot* {
        if (b.slots[0].tag == tag) return &b.slots[0];
        if (b.slots[1].tag == tag) return &b.slots[1];
        if (b.slots[2].tag == tag) return &b.slots[2];

        int victim = 0;
        if (b.slots[0].tag == 0u) {
            victim = 0;
        } else if (b.slots[1].tag == 0u) {
            victim = 1;
        } else if (b.slots[2].tag == 0u) {
            victim = 2;
        } else {
            const int p0 = st.priority[b.slots[0].states[0]] + (b.slots[0].run_cnt >> 1);
            const int p1 = st.priority[b.slots[1].states[0]] + (b.slots[1].run_cnt >> 1);
            const int p2 = st.priority[b.slots[2].states[0]] + (b.slots[2].run_cnt >> 1);
            victim = (p0 <= p1) ? ((p0 <= p2) ? 0 : 2) : ((p1 <= p2) ? 1 : 2);
        }
        auto& s = b.slots[victim];
        s.tag = tag;
        s.run_byte = 0u;
        s.run_cnt = 0u;
        std::memset(s.states, 0, sizeof(s.states));
        return &s;
    }

    std::size_t num_models_;
    std::vector<CmBucket> buckets_;
    std::size_t mask_ = 0;
    StateMap sm_;

    std::vector<std::uint64_t> hashes_;
    std::vector<CmBucket::Slot*> byte_slots_;
    std::vector<std::uint16_t> byte_tags_;
    std::vector<CmBucket::Slot*> cur_slots_;
    std::vector<std::uint8_t> run_bytes_;
    std::vector<std::uint8_t> run_cnts_;
    std::vector<std::uint8_t> run_ok_;
    std::vector<std::uint8_t> last_states_;
};

// ---------------------------------------------------------------------------
// 4-Way Associative Long-Range Dual Match Model with 1-Byte Edit Recovery
// ---------------------------------------------------------------------------
class MatchModel {
  public:
    struct Bucket4 {
        std::uint32_t p[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    };

    MatchModel(const std::uint8_t* buf, std::size_t hash_bytes, bool deep)
        : buf_(buf),
          deep_(deep),
          pm_primary_(16u * 8u * 2u, 255u),
          pm_secondary_(8u * 8u * 2u, 255u),
          pm_recovery_(8u * 8u * 2u, 255u) {
        constexpr double kLenMid[16] = {
            3.5, 4.5, 6.5, 9.5, 13.5, 19.5, 27.5, 39.5,
            55.5, 79.5, 111.5, 191.5, 383.5, 767.5, 2048.0, 8192.0
        };
        for (std::size_t lb = 0; lb < 16u; ++lb) {
            const double p_match = 1.0 - 1.0 / (2.0 * kLenMid[lb] + 2.0);
            for (std::size_t bp = 0; bp < 8u; ++bp) {
                pm_primary_.set_prior(((lb << 3) | bp) * 2u + 0u, 1.0 - p_match, 4u);
                pm_primary_.set_prior(((lb << 3) | bp) * 2u + 1u, p_match, 4u);
                if (lb < 8u) {
                    const double p2 = 1.0 - 1.0 / (kLenMid[lb] + 3.0);
                    pm_secondary_.set_prior(((lb << 3) | bp) * 2u + 0u, 1.0 - p2, 4u);
                    pm_secondary_.set_prior(((lb << 3) | bp) * 2u + 1u, p2, 4u);
                    pm_recovery_.set_prior(((lb << 3) | bp) * 2u + 0u, 1.0 - p2, 4u);
                    pm_recovery_.set_prior(((lb << 3) | bp) * 2u + 1u, p2, 4u);
                }
            }
        }

        // 4-way buckets for ht6 & ht8 (16 B each), 1-way for ht4 & ht12 (4 B each).
        const std::size_t b4 = std::bit_floor(std::max<std::size_t>(1024u, (hash_bytes / 6u) / 4u));
        const std::size_t b6 = std::bit_floor(std::max<std::size_t>(1024u, (hash_bytes / 3u) / sizeof(Bucket4)));
        const std::size_t b8 = std::bit_floor(std::max<std::size_t>(1024u, (hash_bytes / 3u) / sizeof(Bucket4)));
        ht4_.assign(b4, kNoMatch);
        ht6_.resize(b6);
        ht8_.resize(b8);
        m4_ = b4 - 1u;
        m6_ = b6 - 1u;
        m8_ = b8 - 1u;
        if (deep_) {
            const std::size_t b12 = std::bit_floor(std::max<std::size_t>(1024u, (hash_bytes / 6u) / 4u));
            ht12_.assign(b12, kNoMatch);
            m12_ = b12 - 1u;
        }
    }

    void on_byte(std::size_t pos) {
        if (pos == 0u) return;
        const std::uint8_t last_b = buf_[pos - 1u];

        if (m1_len_ > 0u) {
            if (buf_[m1_pos_] == last_b) {
                ++m1_pos_;
                if (m1_len_ < 65535u) ++m1_len_;
                rec_len_ = 0u;
            } else {
                if (m1_len_ >= 6u && m1_pos_ + 1u < pos) {
                    rec_pos_ = m1_pos_ + 1u;
                    rec_len_ = std::min<std::uint32_t>(m1_len_, 255u);
                } else {
                    rec_len_ = 0u;
                }
                m1_len_ = 0u;
            }
        } else if (rec_len_ > 0u) {
            if (buf_[rec_pos_] == last_b && rec_pos_ + 1u < pos) {
                m1_pos_ = rec_pos_ + 1u;
                m1_len_ = std::min<std::uint32_t>(rec_len_ + 1u, 65535u);
                rec_len_ = 0u;
            } else {
                rec_len_ = 0u;
            }
        }

        if (m2_len_ > 0u) {
            if (buf_[m2_pos_] == last_b) {
                ++m2_pos_;
                if (m2_len_ < 65535u) ++m2_len_;
            } else {
                m2_len_ = 0u;
            }
        }

        auto verify_len = [&](std::uint32_t cand, std::uint32_t min_l) -> std::uint32_t {
            if (cand == kNoMatch || cand >= pos || cand < min_l) return 0u;
            if (buf_[cand - 1u] != buf_[pos - 1u] ||
                buf_[cand - 2u] != buf_[pos - 2u] ||
                buf_[cand - 3u] != buf_[pos - 3u] ||
                buf_[cand - 4u] != buf_[pos - 4u]) {
                return 0u;
            }
            std::uint32_t l = 4u;
            const std::uint32_t max_back = std::min<std::uint32_t>(cand, 64u);
            while (l < max_back && buf_[cand - 1u - l] == buf_[pos - 1u - l]) {
                ++l;
            }
            return (l >= min_l) ? l : 0u;
        };

        auto consider = [&](std::uint32_t cand, std::uint32_t min_l) {
            if (cand == kNoMatch) return;
            if (m1_len_ > 0u && cand == m1_pos_) return;
            const std::uint32_t l = verify_len(cand, min_l);
            if (l == 0u) return;
            if (l > m1_len_) {
                if (m1_len_ > m2_len_) {
                    m2_pos_ = m1_pos_;
                    m2_len_ = m1_len_;
                }
                m1_pos_ = cand;
                m1_len_ = l;
            } else if (l > m2_len_ && cand != m1_pos_) {
                m2_pos_ = cand;
                m2_len_ = l;
            }
        };

        std::uint32_t cand12 = kNoMatch;
        std::uint32_t cand4 = kNoMatch;
        Bucket4 prev8{};
        Bucket4 prev6{};
        bool has8 = false, has6 = false;

        if (deep_ && pos >= 12u) {
            const std::uint64_t h12 = hash_combine(
                load_u64(buf_ + pos - 12u), load_u32(buf_ + pos - 4u));
            const std::size_t idx = h12 & m12_;
            cand12 = ht12_[idx];
            ht12_[idx] = static_cast<std::uint32_t>(pos);
        }
        if (pos >= 8u) {
            const std::size_t idx = mix64(load_u64(buf_ + pos - 8u)) & m8_;
            prev8 = ht8_[idx];
            has8 = true;
            auto& b = ht8_[idx];
            b.p[3] = prev8.p[2]; b.p[2] = prev8.p[1]; b.p[1] = prev8.p[0];
            b.p[0] = static_cast<std::uint32_t>(pos);
        }
        if (pos >= 6u) {
            const std::uint64_t v6 = static_cast<std::uint64_t>(load_u32(buf_ + pos - 6u)) |
                                     (static_cast<std::uint64_t>(load_u16(buf_ + pos - 2u)) << 32);
            const std::size_t idx = mix64(v6) & m6_;
            prev6 = ht6_[idx];
            has6 = true;
            auto& b = ht6_[idx];
            b.p[3] = prev6.p[2]; b.p[2] = prev6.p[1]; b.p[1] = prev6.p[0];
            b.p[0] = static_cast<std::uint32_t>(pos);
        }
        if (pos >= 4u) {
            const std::size_t idx = mix64(load_u32(buf_ + pos - 4u)) & m4_;
            cand4 = ht4_[idx];
            ht4_[idx] = static_cast<std::uint32_t>(pos);
        }

        if (m1_len_ < 16u) {
            if (deep_) consider(cand12, 8u);
            if (has8) {
                for (std::uint32_t c : prev8.p) consider(c, 6u);
            }
            if (has6 && m1_len_ < 12u) {
                for (std::uint32_t c : prev6.p) consider(c, 5u);
            }
            if (m1_len_ < 6u) consider(cand4, 4u);
        } else if (m2_len_ == 0u && has8) {
            consider(prev8.p[0], 6u);
            consider(prev8.p[1], 6u);
        }

        m1_ok_ = (m1_len_ > 0u);
        m2_ok_ = (m2_len_ > 0u);
        rec_ok_ = (rec_len_ > 0u);
        m1_byte_ = m1_ok_ ? buf_[m1_pos_] : 0u;
        m2_byte_ = m2_ok_ ? buf_[m2_pos_] : 0u;
        rec_byte_ = rec_ok_ ? buf_[rec_pos_] : 0u;
    }

    void predict(int bpos, std::int16_t* out, std::size_t& out_len) {
        last_m1_idx_ = kNoIdx;
        last_m2_idx_ = kNoIdx;
        last_rec_idx_ = kNoIdx;

        if (m1_ok_) {
            const int eb = (m1_byte_ >> (7 - bpos)) & 1;
            const std::uint32_t lb = len_bucket(m1_len_);
            const std::size_t idx =
                ((static_cast<std::size_t>(lb) << 3) | static_cast<std::size_t>(bpos)) * 2u +
                static_cast<std::size_t>(eb);
            last_m1_idx_ = idx;
            out[out_len++] = stretch_p12(pm_primary_.predict(idx));

            const std::uint32_t capped = std::min<std::uint32_t>(m1_len_, 255u);
            const std::int16_t lo = prob_tables().run_logit[capped];
            out[out_len++] = eb ? lo : static_cast<std::int16_t>(-lo);
        } else {
            out[out_len++] = 0;
            out[out_len++] = 0;
        }

        if (m2_ok_) {
            const int eb = (m2_byte_ >> (7 - bpos)) & 1;
            const std::uint32_t lb = std::min<std::uint32_t>(len_bucket(m2_len_), 7u);
            const std::size_t idx =
                ((static_cast<std::size_t>(lb) << 3) | static_cast<std::size_t>(bpos)) * 2u +
                static_cast<std::size_t>(eb);
            last_m2_idx_ = idx;
            out[out_len++] = stretch_p12(pm_secondary_.predict(idx));
        } else {
            out[out_len++] = 0;
        }

        if (rec_ok_) {
            const int eb = (rec_byte_ >> (7 - bpos)) & 1;
            const std::uint32_t lb = std::min<std::uint32_t>(len_bucket(rec_len_), 7u);
            const std::size_t idx =
                ((static_cast<std::size_t>(lb) << 3) | static_cast<std::size_t>(bpos)) * 2u +
                static_cast<std::size_t>(eb);
            last_rec_idx_ = idx;
            out[out_len++] = stretch_p12(pm_recovery_.predict(idx));
        } else {
            out[out_len++] = 0;
        }
    }

    void update_bit(int bpos, int bit) {
        if (last_m1_idx_ != kNoIdx) pm_primary_.update(last_m1_idx_, bit);
        if (last_m2_idx_ != kNoIdx) pm_secondary_.update(last_m2_idx_, bit);
        if (last_rec_idx_ != kNoIdx) pm_recovery_.update(last_rec_idx_, bit);

        if (m1_ok_ && (((m1_byte_ >> (7 - bpos)) & 1) != bit)) m1_ok_ = false;
        if (m2_ok_ && (((m2_byte_ >> (7 - bpos)) & 1) != bit)) m2_ok_ = false;
        if (rec_ok_ && (((rec_byte_ >> (7 - bpos)) & 1) != bit)) rec_ok_ = false;
    }

    auto mixer_bucket() const -> std::uint32_t {
        if (!m1_ok_) return 0u;
        if (m1_len_ < 6u) return 1u;
        if (m1_len_ < 16u) return 2u;
        return 3u;
    }

  private:
    static constexpr std::uint32_t kNoMatch = 0xFFFFFFFFu;
    static constexpr std::size_t kNoIdx = std::numeric_limits<std::size_t>::max();

    static auto len_bucket(std::uint32_t l) -> std::uint32_t {
        if (l < 4u) return 0u;
        if (l < 6u) return 1u;
        if (l < 8u) return 2u;
        if (l < 12u) return 3u;
        if (l < 16u) return 4u;
        if (l < 24u) return 5u;
        if (l < 32u) return 6u;
        if (l < 48u) return 7u;
        if (l < 64u) return 8u;
        if (l < 96u) return 9u;
        if (l < 128u) return 10u;
        if (l < 256u) return 11u;
        if (l < 512u) return 12u;
        if (l < 1024u) return 13u;
        if (l < 4096u) return 14u;
        return 15u;
    }

    const std::uint8_t* buf_;
    bool deep_;
    std::vector<std::uint32_t> ht4_;
    std::vector<Bucket4> ht6_;
    std::vector<Bucket4> ht8_;
    std::vector<std::uint32_t> ht12_;
    std::size_t m4_ = 0, m6_ = 0, m8_ = 0, m12_ = 0;

    std::uint32_t m1_pos_ = 0, m1_len_ = 0;
    std::uint32_t m2_pos_ = 0, m2_len_ = 0;
    std::uint32_t rec_pos_ = 0, rec_len_ = 0;

    bool m1_ok_ = false, m2_ok_ = false, rec_ok_ = false;
    std::uint8_t m1_byte_ = 0, m2_byte_ = 0, rec_byte_ = 0;

    FlatProbMap pm_primary_;
    FlatProbMap pm_secondary_;
    FlatProbMap pm_recovery_;
    std::size_t last_m1_idx_ = kNoIdx;
    std::size_t last_m2_idx_ = kNoIdx;
    std::size_t last_rec_idx_ = kNoIdx;
};

// ---------------------------------------------------------------------------
// Record / Stride, Word / XML, Indirect, and x86/x64 Exe Trackers
// ---------------------------------------------------------------------------
class RecordTracker {
  public:
    RecordTracker() {
        cpos1_.fill(0u);
        cpos2_.fill(0u);
        rlen_.fill(2u);
        rcnt_.fill(0u);
    }

    void update(std::size_t pos, std::uint8_t c1, std::uint8_t c2) {
        const std::uint32_t p = static_cast<std::uint32_t>(pos);
        const std::uint32_t c12 = (static_cast<std::uint32_t>(c2) << 8) | c1;
        const std::uint32_t prev = cpos1_[c12];
        if (prev != 0u && p > prev) {
            const std::uint32_t d = p - prev;
            if (d >= 2u && d <= 1024u) {
                const std::uint32_t prev2 = cpos2_[c12];
                if (prev2 != 0u && (prev - prev2) == d) {
                    if (d == rlen_[0]) {
                        rcnt_[0] = std::min<std::uint32_t>(rcnt_[0] + 2u, 255u);
                    } else if (d == rlen_[1]) {
                        rcnt_[1] = std::min<std::uint32_t>(rcnt_[1] + 2u, 255u);
                        if (rcnt_[1] > rcnt_[0]) {
                            std::swap(rlen_[0], rlen_[1]);
                            std::swap(rcnt_[0], rcnt_[1]);
                        }
                    } else {
                        if (rcnt_[1] > 0u) --rcnt_[1];
                        if (rcnt_[1] == 0u) {
                            rlen_[1] = d;
                            rcnt_[1] = 2u;
                        }
                    }
                }
            }
        }
        cpos2_[c12] = prev;
        cpos1_[c12] = p;
    }

    auto r1() const -> std::uint32_t { return rlen_[0]; }

  private:
    std::array<std::uint32_t, 65536> cpos1_{};
    std::array<std::uint32_t, 65536> cpos2_{};
    std::array<std::uint32_t, 2> rlen_{};
    std::array<std::uint32_t, 2> rcnt_{};
};

class WordTracker {
  public:
    void update(std::uint8_t b, bool wrt_mode) {
        if (b == '\n' || b == '\r') {
            col_ = 0u;
            indent_ = 0u;
            in_indent_ = true;
        } else {
            ++col_;
            if (in_indent_) {
                if (b == ' ' || b == '\t') {
                    ++indent_;
                } else {
                    in_indent_ = false;
                }
            }
        }

        if (b == '<') {
            in_tag_ = true;
            cur_tag_ = 0x123456789ABCDEF0ULL;
            tag_len_ = 0u;
        } else if (in_tag_) {
            if (b == '>' || b == ' ' || b == '\n' || tag_len_ >= 24u) {
                if (tag_len_ > 0u) last_tag_ = cur_tag_;
                in_tag_ = false;
            } else {
                cur_tag_ = cur_tag_ * 1315423911ULL + b;
                ++tag_len_;
            }
        }

        // Exact WRT multi-byte token boundary tracking (critical with implicit spaces!)
        if (wrt_mode) {
            if (wrt_esc_) {
                wrt_esc_ = false;
                flush_word();
                return;
            }
            if (wrt_rem_ > 0u) {
                w0_ = (w0_ * 0x100000001B3ULL) ^ b;
                ++wlen_;
                if (--wrt_rem_ == 0u) {
                    flush_word();
                }
                return;
            }
            if (b == 0x80u) {
                flush_word();
                wrt_esc_ = true;
                return;
            }
            if (b == 0x81u || b == 0x82u) {
                flush_word();
                return;
            }
            if (b >= 0x83u && b < 0xC3u) {
                flush_word();
                w2_ = w1_;
                w1_ = (0xCBF29CE484222325ULL * 0x100000001B3ULL) ^ b;
                return;
            }
            if (b >= 0xC3u && b < 0xFBu) {
                flush_word();
                w0_ = (0xCBF29CE484222325ULL * 0x100000001B3ULL) ^ b;
                wlen_ = 1u;
                wrt_rem_ = 1u;
                return;
            }
            if (b >= 0xFBu && b < 0xFFu) {
                flush_word();
                w0_ = (0xCBF29CE484222325ULL * 0x100000001B3ULL) ^ b;
                wlen_ = 1u;
                wrt_rem_ = 2u;
                return;
            }
        }

        const bool is_alpha = (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') ||
                              (!wrt_mode && b >= 128u) || b == '_';
        if (is_alpha) {
            const std::uint8_t lb = (b >= 'A' && b <= 'Z')
                                        ? static_cast<std::uint8_t>(b + ('a' - 'A'))
                                        : b;
            w0_ = (w0_ * 0x100000001B3ULL) ^ lb;
            if (wlen_ < 255u) ++wlen_;
        } else {
            flush_word();
        }
    }

    auto w0() const -> std::uint64_t { return w0_; }
    auto w1() const -> std::uint64_t { return w1_; }
    auto w2() const -> std::uint64_t { return w2_; }
    auto last_tag() const -> std::uint64_t { return last_tag_; }
    auto col_bucket() const -> std::uint32_t {
        if (col_ < 8u) return col_;
        if (col_ < 16u) return 8u + ((col_ - 8u) >> 1);
        if (col_ < 64u) return 12u + ((col_ - 16u) >> 3);
        return 18u + std::min<std::uint32_t>((col_ - 64u) >> 5, 13u);
    }
    auto indent() const -> std::uint32_t { return std::min<std::uint32_t>(indent_, 31u); }

  private:
    void flush_word() {
        if (wlen_ > 0u) {
            w2_ = w1_;
            w1_ = w0_;
            w0_ = 0xCBF29CE484222325ULL;
            wlen_ = 0u;
        }
    }

    std::uint64_t w0_ = 0xCBF29CE484222325ULL;
    std::uint64_t w1_ = 0;
    std::uint64_t w2_ = 0;
    std::uint64_t cur_tag_ = 0;
    std::uint64_t last_tag_ = 0;
    std::uint32_t col_ = 0;
    std::uint32_t indent_ = 0;
    std::uint32_t tag_len_ = 0;
    std::uint32_t wlen_ = 0;
    std::uint8_t wrt_rem_ = 0;
    bool wrt_esc_ = false;
    bool in_indent_ = true;
    bool in_tag_ = false;
};

class IndirectTracker {
  public:
    IndirectTracker() : h_o1_(256u, 0u), h_o2_(65536u, 0u) {}

    void update(std::uint8_t b, std::uint8_t c1, std::uint8_t c2) {
        h_o1_[c1] = (h_o1_[c1] << 8) | b;
        h_o2_[(static_cast<std::uint32_t>(c2) << 8) | c1] =
            (h_o2_[(static_cast<std::uint32_t>(c2) << 8) | c1] << 8) | b;
    }

    auto get_o1(std::uint8_t c1) const -> std::uint32_t { return h_o1_[c1]; }
    auto get_o2(std::uint8_t c1, std::uint8_t c2) const -> std::uint32_t {
        return h_o2_[(static_cast<std::uint32_t>(c2) << 8) | c1];
    }

  private:
    std::vector<std::uint32_t> h_o1_;
    std::vector<std::uint32_t> h_o2_;
};

// Lightweight x86/x64 instruction boundary & opcode state tracker
class ExeTracker {
  public:
    void update(std::uint8_t b) {
        if (rem_ > 0u) {
            --rem_;
            ++op_off_;
            return;
        }
        // Recognize common x86/x64 instruction prefixes & opcodes
        if ((b & 0xF0u) == 0x40u) {
            // REX prefix: keep waiting for main opcode
            rex_ = b;
            return;
        }
        prev_op_ = cur_op_;
        cur_op_ = b;
        op_off_ = 1u;
        if (b == 0xE8u || b == 0xE9u || b == 0x68u) {
            rem_ = 4u;
        } else if (b == 0x0Fu) {
            rem_ = 5u;
        } else if (b == 0x8Bu || b == 0x89u || b == 0x8Du || b == 0x83u ||
                   b == 0x3Bu || b == 0x39u || b == 0x85u || b == 0x01u ||
                   b == 0x29u || b == 0x33u || b == 0x31u) {
            rem_ = 2u;
        } else if (b == 0xC7u || b == 0x81u) {
            rem_ = 5u;
        } else if (b == 0xEBu || (b >= 0x70u && b <= 0x7Fu)) {
            rem_ = 1u;
        } else {
            rem_ = 0u;
        }
    }

    auto ctx1(std::uint8_t c1) const -> std::uint64_t {
        return (static_cast<std::uint64_t>(cur_op_) << 16) |
               (static_cast<std::uint64_t>(op_off_ & 7u) << 8) | c1;
    }
    auto ctx2() const -> std::uint64_t {
        return (static_cast<std::uint64_t>(prev_op_) << 16) |
               (static_cast<std::uint64_t>(cur_op_) << 8) |
               (static_cast<std::uint64_t>(op_off_ & 7u) << 4) | (rex_ & 0xFu);
    }

  private:
    std::uint8_t cur_op_ = 0;
    std::uint8_t prev_op_ = 0;
    std::uint8_t rex_ = 0;
    std::uint8_t op_off_ = 0;
    std::uint8_t rem_ = 0;
};

// ---------------------------------------------------------------------------
// Exact Direct Order-0 / Order-1 / Order-2 Models with bpos-Indexed StateMaps
// ---------------------------------------------------------------------------
class DirectModels {
  public:
    explicit DirectModels(bool use_o2)
        : use_o2_(use_o2),
          o0_(256u, 0u),
          o1_(256u * 256u, 0u),
          o15_(16u * 256u * 256u, 0u),
          sm0_(8u, 1023u),
          sm1_slow_(8u, 511u),
          sm1_fast_(8u, 63u),
          sm15_(8u, 511u),
          sm2_slow_(8u, 511u),
          sm2_fast_(8u, 63u) {
        if (use_o2_) {
            o2_.assign(65536u * 256u, 0u);
        }
    }

    void set_byte_ctx(std::uint8_t c1, std::uint8_t c2) {
        o1_base_ = static_cast<std::size_t>(c1) << 8;
        o15_base_ = ((static_cast<std::size_t>(c2 >> 4) << 8) | c1) << 8;
        o2_base_ = ((static_cast<std::size_t>(c2) << 8) | c1) << 8;
        prefetch_l1(&o15_[o15_base_]);
        if (use_o2_) {
            prefetch_l1(&o2_[o2_base_]);
        }
    }

    // Outputs 6 features: O0, O1_slow, O1_fast, O1.5(c2_hi,c1), O2_slow, O2_fast
    void predict(int bpos, std::uint32_t c0, std::int16_t* out, std::size_t& out_len) {
        const std::size_t bg = static_cast<std::size_t>(bpos) << 8;
        s0_ = o0_[c0];
        s1_ = o1_[o1_base_ + c0];
        s15_ = o15_[o15_base_ + c0];
        out[out_len++] = stretch_p12(sm0_.predict(bg + s0_));
        out[out_len++] = (s1_ == 0u) ? std::int16_t{0} : stretch_p12(sm1_slow_.predict(bg + s1_));
        out[out_len++] = (s1_ == 0u) ? std::int16_t{0} : stretch_p12(sm1_fast_.predict(bg + s1_));
        out[out_len++] = (s15_ == 0u) ? std::int16_t{0} : stretch_p12(sm15_.predict(bg + s15_));
        if (use_o2_) {
            s2_ = o2_[o2_base_ + c0];
            out[out_len++] = (s2_ == 0u) ? std::int16_t{0} : stretch_p12(sm2_slow_.predict(bg + s2_));
            out[out_len++] = (s2_ == 0u) ? std::int16_t{0} : stretch_p12(sm2_fast_.predict(bg + s2_));
        } else {
            out[out_len++] = 0;
            out[out_len++] = 0;
        }
    }

    void update(int bpos, std::uint32_t c0, int bit) {
        const std::size_t bg = static_cast<std::size_t>(bpos) << 8;
        const auto& st = state_table();
        sm0_.update(bg + s0_, bit);
        o0_[c0] = st.nex[s0_][static_cast<std::size_t>(bit)];

        if (s1_ != 0u) {
            sm1_slow_.update(bg + s1_, bit);
            sm1_fast_.update(bg + s1_, bit);
        }
        o1_[o1_base_ + c0] = st.nex[s1_][static_cast<std::size_t>(bit)];

        if (s15_ != 0u) sm15_.update(bg + s15_, bit);
        o15_[o15_base_ + c0] = st.nex[s15_][static_cast<std::size_t>(bit)];

        if (use_o2_) {
            if (s2_ != 0u) {
                sm2_slow_.update(bg + s2_, bit);
                sm2_fast_.update(bg + s2_, bit);
            }
            o2_[o2_base_ + c0] = st.nex[s2_][static_cast<std::size_t>(bit)];
        }
    }

  private:
    bool use_o2_;
    std::vector<std::uint8_t> o0_;
    std::vector<std::uint8_t> o1_;
    std::vector<std::uint8_t> o15_;
    std::vector<std::uint8_t> o2_;
    StateMap sm0_;
    StateMap sm1_slow_;
    StateMap sm1_fast_;
    StateMap sm15_;
    StateMap sm2_slow_;
    StateMap sm2_fast_;

    std::size_t o1_base_ = 0;
    std::size_t o15_base_ = 0;
    std::size_t o2_base_ = 0;
    std::uint8_t s0_ = 0, s1_ = 0, s15_ = 0, s2_ = 0;
};

// ---------------------------------------------------------------------------
// Bit-Exact SIMD (AVX2 / Scalar) Logistic Mixer
// ---------------------------------------------------------------------------
class LogisticMixer {
  public:
    LogisticMixer(std::size_t num_inputs, std::size_t num_contexts,
                  std::int16_t init_weight = 120)
        : n_((num_inputs + 15u) & ~std::size_t{15}),
          num_ctx_(num_contexts),
          weights_(num_contexts * n_, init_weight),
          counts_(num_contexts, 0u),
          avx2_(has_avx2()) {}

    auto predict(const std::int16_t* x, std::size_t ctx) -> std::int32_t {
        last_ctx_ = ctx;
        const std::int16_t* w = weights_.data() + ctx * n_;
        const std::int32_t sum = dot_product(x, w, n_, avx2_);
        last_out_ = std::clamp(sum >> 12, -2047, 2047);
        last_p12_ = squash_to_12(last_out_);
        return last_out_;
    }

    void update(const std::int16_t* x, int bit, int min_lr = 2, int max_lr = 12) {
        const std::uint32_t cnt = counts_[last_ctx_];
        if (cnt < 65535u) ++counts_[last_ctx_];
        int lr = min_lr;
        if (cnt < 128u) {
            lr = max_lr - static_cast<int>((cnt * static_cast<std::uint32_t>(max_lr - min_lr)) >> 7);
        }
        const std::int32_t err = ((bit << 12) - static_cast<std::int32_t>(last_p12_)) * lr;
        const std::int16_t err16 = static_cast<std::int16_t>(std::clamp(err >> 3, -32767, 32767));
        std::int16_t* w = weights_.data() + last_ctx_ * n_;
        train_weights(x, w, n_, err16, avx2_);
    }

  private:
    static auto dot_product(const std::int16_t* x, const std::int16_t* w,
                            std::size_t n, bool use_avx2) -> std::int32_t {
#if FZAC_X64
        if (use_avx2) {
            __m256i acc = _mm256_setzero_si256();
            for (std::size_t i = 0; i < n; i += 16u) {
                const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
                const __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(vx, vw));
            }
            const __m128i lo = _mm256_castsi256_si128(acc);
            const __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i s = _mm_add_epi32(lo, hi);
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            return _mm_cvtsi128_si32(s);
        }
#else
        (void)use_avx2;
#endif
        std::int32_t sum = 0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += static_cast<std::int32_t>(x[i]) * static_cast<std::int32_t>(w[i]);
        }
        return sum;
    }

    static void train_weights(const std::int16_t* x, std::int16_t* w,
                              std::size_t n, std::int16_t err16, bool use_avx2) {
#if FZAC_X64
        if (use_avx2) {
            const __m256i verr = _mm256_set1_epi16(err16);
            const __m256i vmin = _mm256_set1_epi16(-30000);
            const __m256i vmax = _mm256_set1_epi16(30000);
            for (std::size_t i = 0; i < n; i += 16u) {
                const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
                const __m256i dw = _mm256_mulhrs_epi16(vx, verr);
                __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
                vw = _mm256_adds_epi16(vw, dw);
                vw = _mm256_max_epi16(vw, vmin);
                vw = _mm256_min_epi16(vw, vmax);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(w + i), vw);
            }
            return;
        }
#else
        (void)use_avx2;
#endif
        for (std::size_t i = 0; i < n; ++i) {
            const std::int32_t dw =
                (static_cast<std::int32_t>(x[i]) * static_cast<std::int32_t>(err16) + 16384) >> 15;
            w[i] = static_cast<std::int16_t>(
                std::clamp(static_cast<std::int32_t>(w[i]) + dw, -30000, 30000));
        }
    }

    std::size_t n_;
    std::size_t num_ctx_;
    std::vector<std::int16_t> weights_;
    std::vector<std::uint16_t> counts_;
    std::size_t last_ctx_ = 0;
    std::int32_t last_out_ = 0;
    std::uint32_t last_p12_ = 2048u;
    bool avx2_ = false;
};

// ---------------------------------------------------------------------------
// Count-Adaptive Probability Map (APM / SSE) with Linear Logit Interpolation
// ---------------------------------------------------------------------------
class APM {
  public:
    explicit APM(std::size_t num_contexts)
        : t_(num_contexts * 33u), cnt_(num_contexts, 0u) {
        for (std::size_t c = 0; c < num_contexts; ++c) {
            for (int j = 0; j < 33; ++j) {
                t_[c * 33u + static_cast<std::size_t>(j)] =
                    static_cast<std::uint16_t>(squash_to_16(j * 128 - 2048));
            }
        }
    }

    auto predict(std::int32_t pr_stretched, std::size_t ctx) -> std::uint32_t {
        const std::int32_t s = std::clamp(pr_stretched, -2047, 2047);
        const std::uint32_t scaled = static_cast<std::uint32_t>(s + 2048) * 32u;
        const std::uint32_t idx = scaled >> 12;
        const std::uint32_t frac = scaled & 4095u;
        last_ctx_ = ctx;
        last_base_ = ctx * 33u + idx;
        last_frac_ = frac;
        const std::uint32_t p0 = t_[last_base_];
        const std::uint32_t p1 = t_[last_base_ + 1u];
        const std::uint32_t p = (p0 * (4096u - frac) + p1 * frac) >> 12;
        return std::clamp<std::uint32_t>(p, 1u, 65535u);
    }

    void update(int bit) {
        std::uint8_t c = cnt_[last_ctx_];
        const int rate = (c < 24u) ? 5 : ((c < 96u) ? 6 : 7);
        if (c < 255u) cnt_[last_ctx_] = static_cast<std::uint8_t>(c + 1u);

        const std::int32_t target = bit ? 65535 : 0;
        const int r0 = (last_frac_ < 2048u) ? rate : (rate + 1);
        const int r1 = (last_frac_ < 2048u) ? (rate + 1) : rate;
        std::int32_t v0 = t_[last_base_];
        std::int32_t v1 = t_[last_base_ + 1u];
        v0 += (target - v0) >> r0;
        v1 += (target - v1) >> r1;
        t_[last_base_] = static_cast<std::uint16_t>(std::clamp(v0, 1, 65535));
        t_[last_base_ + 1u] = static_cast<std::uint16_t>(std::clamp(v1, 1, 65535));
    }

  private:
    std::vector<std::uint16_t> t_;
    std::vector<std::uint8_t> cnt_;
    std::size_t last_ctx_ = 0;
    std::size_t last_base_ = 0;
    std::uint32_t last_frac_ = 0;
};

// ---------------------------------------------------------------------------
// Reversible x86/x64 BCJ Filter (E8/E9 + Optional 0F 80..8F Jcc)
// ---------------------------------------------------------------------------
inline auto is_bcj_candidate(std::uint32_t v) -> bool {
    return ((v + 0x01000000u) & 0xFE000000u) == 0u;
}

inline auto sign_extend_25(std::uint32_t u) -> std::uint32_t {
    u &= 0x01FFFFFFu;
    return (u & 0x01000000u) ? (u | 0xFF000000u) : u;
}

auto detect_bcj(const std::uint8_t* data, std::size_t n, bool& enable_jcc) -> bool {
    enable_jcc = false;
    if (n < 64u) return false;
    std::array<std::uint32_t, 1024> raw_seen{};
    std::array<std::uint32_t, 1024> abs_seen{};
    raw_seen.fill(0xFFFFFFFFu);
    abs_seen.fill(0xFFFFFFFFu);

    std::uint32_t raw_dups = 0, abs_dups = 0;
    std::uint32_t jcc_raw = 0, jcc_abs = 0;
    std::size_t last_unconv = 0;
    bool has_unconv = false;
    const std::size_t limit = std::min<std::size_t>(n - 6u, 4u << 20);
    for (std::size_t i = 0; i <= limit; ++i) {
        if (data[i] == 0xE8u || data[i] == 0xE9u) {
            if (has_unconv && (i - last_unconv) <= 3u) {
                last_unconv = i;
                continue;
            }
            const std::uint32_t rel = load_u32(data + i + 1u);
            if (!is_bcj_candidate(rel)) {
                last_unconv = i;
                has_unconv = true;
                continue;
            }
            const std::uint32_t ip = static_cast<std::uint32_t>(i + 5u);
            const std::uint32_t av = sign_extend_25(rel + ip);
            const std::uint32_t hr = (rel * 2654435761u) >> 22;
            if (raw_seen[hr] == rel) ++raw_dups; else raw_seen[hr] = rel;
            const std::uint32_t ha = (av * 2654435761u) >> 22;
            if (abs_seen[ha] == av) ++abs_dups; else abs_seen[ha] = av;
            i += 4u;
        } else if (data[i] == 0x0Fu && (data[i + 1u] & 0xF0u) == 0x80u) {
            const std::uint32_t rel = load_u32(data + i + 2u);
            if (is_bcj_candidate(rel)) {
                const std::uint32_t ip = static_cast<std::uint32_t>(i + 6u);
                const std::uint32_t av = sign_extend_25(rel + ip);
                const std::uint32_t hr = (rel * 2654435761u) >> 22;
                if (raw_seen[hr] == rel) ++jcc_raw; else raw_seen[hr] = rel;
                const std::uint32_t ha = (av * 2654435761u) >> 22;
                if (abs_seen[ha] == av) ++jcc_abs; else abs_seen[ha] = av;
                i += 5u;
            }
        }
    }
    const bool use_e89 = (abs_dups >= 16u && abs_dups > raw_dups);
    if (use_e89 && jcc_abs >= 16u && jcc_abs > jcc_raw) {
        enable_jcc = true;
    }
    return use_e89;
}

void bcj_e8e9_forward(std::uint8_t* data, std::size_t n) {
    if (n < 5u) return;
    std::size_t last_unconv = 0;
    bool has_unconv = false;
    for (std::size_t i = 0; i + 5u <= n; ++i) {
        if (data[i] != 0xE8u && data[i] != 0xE9u) continue;
        if (has_unconv && (i - last_unconv) <= 3u) {
            last_unconv = i;
            continue;
        }
        const std::uint32_t rel = load_u32(data + i + 1u);
        if (!is_bcj_candidate(rel)) {
            last_unconv = i;
            has_unconv = true;
            continue;
        }
        const std::uint32_t ip = static_cast<std::uint32_t>(i + 5u);
        store_u32(data + i + 1u, sign_extend_25(rel + ip));
        i += 4u;
    }
}

void bcj_e8e9_inverse(std::uint8_t* data, std::size_t n) {
    if (n < 5u) return;
    std::size_t last_unconv = 0;
    bool has_unconv = false;
    for (std::size_t i = 0; i + 5u <= n; ++i) {
        if (data[i] != 0xE8u && data[i] != 0xE9u) continue;
        if (has_unconv && (i - last_unconv) <= 3u) {
            last_unconv = i;
            continue;
        }
        const std::uint32_t av = load_u32(data + i + 1u);
        if (!is_bcj_candidate(av)) {
            last_unconv = i;
            has_unconv = true;
            continue;
        }
        const std::uint32_t ip = static_cast<std::uint32_t>(i + 5u);
        store_u32(data + i + 1u, sign_extend_25(av - ip));
        i += 4u;
    }
}

void bcj_jcc_forward(std::uint8_t* data, std::size_t n) {
    if (n < 6u) return;
    std::size_t last_unconv = 0;
    bool has_unconv = false;
    for (std::size_t i = 0; i + 6u <= n; ++i) {
        if (data[i] != 0x0Fu || (data[i + 1u] & 0xF0u) != 0x80u) continue;
        if (has_unconv && (i - last_unconv) <= 4u) {
            last_unconv = i;
            continue;
        }
        const std::uint32_t rel = load_u32(data + i + 2u);
        if (!is_bcj_candidate(rel)) {
            last_unconv = i;
            has_unconv = true;
            continue;
        }
        const std::uint32_t ip = static_cast<std::uint32_t>(i + 6u);
        store_u32(data + i + 2u, sign_extend_25(rel + ip));
        i += 5u;
    }
}

void bcj_jcc_inverse(std::uint8_t* data, std::size_t n) {
    if (n < 6u) return;
    std::size_t last_unconv = 0;
    bool has_unconv = false;
    for (std::size_t i = 0; i + 6u <= n; ++i) {
        if (data[i] != 0x0Fu || (data[i + 1u] & 0xF0u) != 0x80u) continue;
        if (has_unconv && (i - last_unconv) <= 4u) {
            last_unconv = i;
            continue;
        }
        const std::uint32_t av = load_u32(data + i + 2u);
        if (!is_bcj_candidate(av)) {
            last_unconv = i;
            has_unconv = true;
            continue;
        }
        const std::uint32_t ip = static_cast<std::uint32_t>(i + 6u);
        store_u32(data + i + 2u, sign_extend_25(av - ip));
        i += 5u;
    }
}

// ---------------------------------------------------------------------------
// Memory Sizing Helper (scales smoothly from 4 MB up to ~8.5 GB for 1 GB input)
// ---------------------------------------------------------------------------
struct MemBudget {
    std::size_t cm_low_bytes;
    std::size_t cm_high_bytes;
    std::size_t match_bytes;
    std::size_t ppm_bytes;
    int ppm_order;
    bool use_direct_o2;
};

auto budget_for_mem_level(std::uint8_t mem_level) -> MemBudget {
    switch (mem_level) {
        case 0: return {1u << 20,   2u << 20,   1u << 20,   2u << 20,   8,  false};
        case 1: return {4u << 20,   8u << 20,   4u << 20,   8u << 20,   10, true};
        case 2: return {16u << 20,  32u << 20,  8u << 20,   24u << 20,  12, true};
        case 3: return {32u << 20,  64u << 20,  16u << 20,  64u << 20,  12, true};
        case 4: return {128u << 20, 256u << 20, 64u << 20,  192u << 20, 14, true};
        case 5: return {256u << 20, 512u << 20, 128u << 20, 384u << 20, 14, true};
        case 6: return {512u << 20, std::size_t{1024} << 20, 384u << 20, 768u << 20, 16, true};
        case 7: return {std::size_t{1024} << 20, std::size_t{2048} << 20, 768u << 20, std::size_t{1280} << 20, 16, true};
        default: return {std::size_t{1536} << 20, std::size_t{3072} << 20, std::size_t{1024} << 20, std::size_t{1792} << 20, 16, true};
    }
}

// ---------------------------------------------------------------------------
// Unified Context Mixing Predictor (48 inputs -> 4 Stage-1 -> Meta-Mixer -> 3 APM)
// ---------------------------------------------------------------------------
class CmPredictor {
  public:
    static constexpr std::size_t kNumLowModels  = 14u;
    static constexpr std::size_t kNumHighModels = 10u;
    static constexpr std::size_t kMixerInputs   = 48u;  // 3 AVX2 registers!

    CmPredictor(const std::uint8_t* buf, const CmConfig& cfg)
        : buf_(buf),
          cfg_(cfg),
          deep_((cfg.flags & kCmDeepOrders) != 0u),
          exe_mode_((cfg.flags & kCmExe) != 0u),
          wrt_mode_(cfg.transform == kTransformWrt),
          use_ppm_((cfg.flags & kCmPPM) != 0u),
          use_lstm_((cfg.flags & kCmLSTM) != 0u),
          budget_(budget_for_mem_level(cfg.mem_level)),
          cm_low_(budget_.cm_low_bytes, kNumLowModels),
          cm_high_(budget_.cm_high_bytes, kNumHighModels),
          direct_(budget_.use_direct_o2),
          match_(buf, budget_.match_bytes, deep_),
          bit_pm_(16384u, 255u),
          mix_c1_(kMixerInputs, 256u * 8u, 110),
          mix_c0_(kMixerInputs, 256u, 110),
          mix_state_(kMixerInputs, 4u * 4u * 8u * 4u, 110),
          mix_cls_(kMixerInputs, 4u * 4u * 8u, 110),
          mix_c12_(kMixerInputs, 16u * 16u * 8u, 110),
          meta_mix_(16u, 8u * 4u * 4u, 340),
          apm1_(256u),
          apm2_(256u * 8u),
          apm3_(4u * 4u * 8u * 2u),
          apm4_(16u * 16u * 8u) {
        if (use_ppm_) {
            ppm_ = std::make_unique<PpmModel>(buf, budget_.ppm_bytes, budget_.ppm_order);
        }
        if (use_lstm_) {
            const std::uint8_t lstm_tier = (cfg.extra >> 1) & 0x03u;
            const std::size_t num_layers = (lstm_tier >= 2u) ? 2u : 1u;
            lstm_ = std::make_unique<LstmModel>(num_layers, 64u, 32u);
        }
        inputs_.fill(0);
        meta_inputs_.fill(0);
    }

    void start_byte(std::size_t pos) {
        pos_ = pos;
        c1_ = (pos >= 1u) ? buf_[pos - 1u] : 0u;
        c2_ = (pos >= 2u) ? buf_[pos - 2u] : 0u;
        c3_ = (pos >= 3u) ? buf_[pos - 3u] : 0u;
        c4_ = (pos >= 4u) ? buf_[pos - 4u] : 0u;
        c5_ = (pos >= 5u) ? buf_[pos - 5u] : 0u;
        c1_cls_ = byte_class(c1_);
        c2_cls_ = byte_class(c2_);

        direct_.set_byte_ctx(c1_, c2_);
        match_.on_byte(pos);
        if (ppm_) ppm_->start_byte(pos);
        if (lstm_) lstm_->start_byte();

        // Single rolling scan over last 15 bytes
        std::array<std::uint64_t, 16> ro{};
        std::uint64_t v = 0xCBF29CE484222325ULL;
        const std::size_t max_o = std::min<std::size_t>(pos, 15u);
        for (std::size_t k = 1u; k <= max_o; ++k) {
            v = (v * 0x100000001B3ULL) ^ buf_[pos - k];
            ro[k] = v;
        }
        for (std::size_t k = max_o + 1u; k <= 15u; ++k) {
            ro[k] = v;
        }

        // --- Low/Medium/Sparse/Indirect/Exe partition (protected from O15 churn!) ---
        std::array<std::uint64_t, kNumLowModels> hl{};
        std::size_t li = 0u;
        hl[li++] = mix64(ro[3] ^ 0x03030303ULL);
        hl[li++] = mix64(ro[4] ^ 0x04040404ULL);
        hl[li++] = mix64(ro[5] ^ 0x05050505ULL);
        // 5 Sparse / Skip contexts
        hl[li++] = mix64(static_cast<std::uint64_t>(c2_) ^ 0x90000000ULL);
        hl[li++] = mix64(((static_cast<std::uint64_t>(c3_) << 8) | c1_) ^ 0x91000001ULL);
        hl[li++] = mix64(((static_cast<std::uint64_t>(c4_) << 8) | c1_) ^ 0x92000002ULL);
        hl[li++] = mix64(((static_cast<std::uint64_t>(c3_) << 8) | c2_) ^ 0x93000003ULL);
        hl[li++] = mix64(((static_cast<std::uint64_t>(c4_) << 16) |
                          (static_cast<std::uint64_t>(c2_) << 8) | c1_) ^ 0x94000004ULL);
        // Word0 + Indirect O1/O2
        hl[li++] = mix64(word_.w0() ^ 0x11111111ULL);
        hl[li++] = mix64(((static_cast<std::uint64_t>(ind_.get_o1(c1_)) << 8) | c1_) ^ 0xB1000001ULL);
        hl[li++] = mix64(((static_cast<std::uint64_t>(ind_.get_o2(c1_, c2_)) << 8) | c1_) ^ 0xB2000002ULL);
        // Record or Exe contexts (3)
        const std::uint32_t r1 = rec_.r1();
        const std::uint8_t br1 = (pos >= r1) ? buf_[pos - r1] : 0u;
        const std::uint8_t br1_2 = (pos >= 2u * r1) ? buf_[pos - 2u * r1] : 0u;
        if (exe_mode_) {
            hl[li++] = mix64(exe_.ctx1(c1_) ^ 0xC1000001ULL);
            hl[li++] = mix64(exe_.ctx2() ^ 0xC2000002ULL);
            hl[li++] = mix64(((static_cast<std::uint64_t>(pos & 3u) << 24) |
                              (static_cast<std::uint64_t>(c3_) << 16) |
                              (static_cast<std::uint64_t>(c2_) << 8) | c1_) ^ 0xC3000003ULL);
        } else {
            hl[li++] = mix64(((static_cast<std::uint64_t>(br1) << 8) | c1_) ^ 0xA1000001ULL);
            hl[li++] = mix64(((static_cast<std::uint64_t>(br1_2) << 16) |
                              (static_cast<std::uint64_t>(br1) << 8) |
                              static_cast<std::uint64_t>(pos % r1)) ^ 0xA2000002ULL);
            hl[li++] = mix64(((static_cast<std::uint64_t>(c5_) << 16) |
                              (static_cast<std::uint64_t>(c3_) << 8) | c1_) ^ 0x95000005ULL);
        }

        // --- High-order & Word-Ngram partition ---
        std::array<std::uint64_t, kNumHighModels> hh{};
        std::size_t hi = 0u;
        hh[hi++] = mix64(ro[6]  ^ 0x06060606ULL);
        hh[hi++] = mix64(ro[7]  ^ 0x07070707ULL);
        hh[hi++] = mix64(ro[8]  ^ 0x08080808ULL);
        hh[hi++] = mix64(ro[10] ^ 0x0A0A0A0AULL);
        hh[hi++] = mix64(ro[12] ^ 0x0C0C0C0CULL);
        hh[hi++] = mix64(ro[15] ^ 0x0F0F0F0FULL);
        hh[hi++] = hash_combine(word_.w1(), word_.w0() ^ 0x22222222ULL);
        hh[hi++] = hash_combine(hash_combine(word_.w2(), word_.w1()), word_.w0() ^ 0x33333333ULL);
        hh[hi++] = hash_combine(word_.last_tag(), word_.w0() ^ 0x44444444ULL);
        hh[hi++] = mix64(((static_cast<std::uint64_t>(word_.col_bucket()) << 24) |
                          (static_cast<std::uint64_t>(word_.indent()) << 16) |
                          (static_cast<std::uint64_t>(c2_) << 8) | c1_) ^ 0x66666666ULL);

        cm_low_.lookup_byte(hl.data());
        cm_high_.lookup_byte(hh.data());

        c0_ = 1u;
        sub_ = 1u;
    }

    auto predict_bit(int bpos) -> std::uint32_t {
        if (bpos == 4) {
            const std::uint32_t hi_nib = c0_ & 0xFu;
            cm_low_.lookup_low_nibble(hi_nib);
            cm_high_.lookup_low_nibble(hi_nib);
            sub_ = 1u;
        }

        std::size_t n_in = 0u;
        // [0..2] Direct exact models (O0, O1, O2 with bpos-indexed StateMaps)
        direct_.predict(bpos, c0_, inputs_.data(), n_in);
        // [3..6] 4-Way MatchModel (primary_map, primary_analytic, secondary, recovery)
        match_.predict(bpos, inputs_.data(), n_in);
        // [7] 14-bit raw bit-history stream context
        const std::size_t bidx =
            ((static_cast<std::size_t>(bpos) & 1u) << 13) | (bit_hist_ & 8191u);
        last_bit_idx_ = bidx;
        inputs_[n_in++] = stretch_p12(bit_pm_.predict(bidx));
        // [8..21] Low/Sparse/Word0/Indirect/Exe ContextMap (14 models)
        std::int16_t run_low = 0;
        const std::uint32_t hits_lo = cm_low_.predict(bpos, sub_, inputs_.data(), n_in, run_low);
        // [22..31] High-Order/Word-Ngram ContextMap (10 models)
        std::int16_t run_hi = 0;
        const std::uint32_t hits_hi = cm_high_.predict(bpos, sub_, inputs_.data(), n_in, run_hi);
        // [32..33] Best run-length logits (low & high partitions)
        inputs_[n_in++] = run_low;
        inputs_[n_in++] = run_hi;
        // [34..35] Variable-order byte-level PPMd (full with exclusion + highest order)
        const std::int16_t ppm_full = ppm_ ? stretch_p12(ppm_->predict_bit(c0_)) : std::int16_t{0};
        const std::int16_t ppm_high = ppm_ ? stretch_p12(ppm_->predict_high_order_bit(c0_)) : std::int16_t{0};
        inputs_[n_in++] = ppm_full;
        inputs_[n_in++] = ppm_high;
        // [36] Online Byte-Level LSTM Neural Predictor!
        const std::int16_t lstm_logit = lstm_ ? stretch_p12(lstm_->predict_bit(c0_)) : std::int16_t{0};
        inputs_[n_in++] = lstm_logit;
        // [37] Constant bias (+1.0 nat)
        inputs_[n_in++] = 256;
        while (n_in < kMixerInputs) inputs_[n_in++] = 0;

        const std::uint32_t mb = match_.mixer_bucket();
        const std::uint32_t hit_b = std::min<std::uint32_t>((hits_lo + hits_hi) >> 2, 3u);
        const std::uint32_t ubpos = static_cast<std::uint32_t>(bpos);

        const std::size_t mctx_c1 = (static_cast<std::size_t>(c1_) << 3) | ubpos;
        const std::size_t mctx_c0 = c0_ & 255u;
        const std::size_t mctx_st = (((static_cast<std::size_t>(mb) << 2) | hit_b) << 5) |
                                    (ubpos << 2) | c1_cls_;
        const std::size_t mctx_cl = (((static_cast<std::size_t>(c2_cls_) << 2) | c1_cls_) << 3) | ubpos;
        const std::size_t mctx_c12 =
            ((static_cast<std::size_t>(c2_ >> 4) << 7) | (static_cast<std::size_t>(c1_ >> 4) << 3) | ubpos);

        // 5 Stage-1 Logistic Mixers -> Stage-2 Meta-Mixer
        meta_inputs_[0] = static_cast<std::int16_t>(mix_c1_.predict(inputs_.data(), mctx_c1));
        meta_inputs_[1] = static_cast<std::int16_t>(mix_c0_.predict(inputs_.data(), mctx_c0));
        meta_inputs_[2] = static_cast<std::int16_t>(mix_state_.predict(inputs_.data(), mctx_st));
        meta_inputs_[3] = static_cast<std::int16_t>(mix_cls_.predict(inputs_.data(), mctx_cl));
        meta_inputs_[4] = static_cast<std::int16_t>(mix_c12_.predict(inputs_.data(), mctx_c12));
        // Direct skip connections for orthogonal anchor paradigms
        meta_inputs_[5]  = inputs_[6];   // match primary map
        meta_inputs_[6]  = inputs_[7];   // match primary analytic
        meta_inputs_[7]  = inputs_[4];   // direct O2 slow
        meta_inputs_[8]  = run_hi;       // high-order run
        meta_inputs_[9]  = ppm_full;     // byte-level PPMd
        meta_inputs_[10] = ppm_high;     // highest-order PPM
        meta_inputs_[11] = lstm_logit;   // online neural LSTM
        meta_inputs_[12] = 256;          // bias

        const std::size_t meta_ctx = ((static_cast<std::size_t>(mb) << 5) |
                                      (ubpos << 2) | c1_cls_);
        const std::int32_t s_mix = meta_mix_.predict(meta_inputs_.data(), meta_ctx);
        const std::uint32_t p_mix16 = squash_to_16(s_mix);

        // 4-Stage Count-Adaptive Cascaded APM / SSE
        const std::uint32_t p_apm1 = apm1_.predict(s_mix, c0_ & 255u);
        const std::uint32_t p1 = (3u * p_apm1 + p_mix16) >> 2;
        const std::int32_t s1 = stretch_p12(p1 >> 4);

        const std::uint32_t p_apm2 = apm2_.predict(s1, mctx_c1);
        const std::uint32_t p2 = (p_apm2 + p1) >> 1;
        const std::int32_t s2 = stretch_p12(p2 >> 4);

        const std::uint32_t p_apm4 = apm4_.predict(s2, mctx_c12);
        const std::uint32_t p3 = (p_apm4 + p2) >> 1;
        const std::int32_t s3 = stretch_p12(p3 >> 4);

        const std::size_t apm3_ctx = (((static_cast<std::size_t>(mb) << 2) | c1_cls_) << 4) |
                                     (ubpos << 1) | last_err_;
        const std::uint32_t p_apm3 = apm3_.predict(s3, apm3_ctx);
        const std::uint32_t p_final = (3u * p_apm3 + p3) >> 2;
        last_p_final_ = std::clamp<std::uint32_t>(p_final, 1u, 65535u);
        return last_p_final_;
    }

    void update_bit(int bpos, int bit) {
        last_err_ = ((bit == 1 && last_p_final_ < 32768u) ||
                     (bit == 0 && last_p_final_ >= 32768u)) ? 1u : 0u;

        apm1_.update(bit);
        apm2_.update(bit);
        apm4_.update(bit);
        apm3_.update(bit);

        meta_mix_.update(meta_inputs_.data(), bit, 2, 10);
        mix_c1_.update(inputs_.data(), bit, 2, 12);
        mix_c0_.update(inputs_.data(), bit, 2, 10);
        mix_state_.update(inputs_.data(), bit, 2, 12);
        mix_cls_.update(inputs_.data(), bit, 2, 10);
        mix_c12_.update(inputs_.data(), bit, 2, 12);

        direct_.update(bpos, c0_, bit);
        match_.update_bit(bpos, bit);
        bit_pm_.update(last_bit_idx_, bit);
        cm_low_.update_bit(bpos, sub_, bit);
        cm_high_.update_bit(bpos, sub_, bit);

        c0_ = (c0_ << 1) | static_cast<std::uint32_t>(bit);
        sub_ = (sub_ << 1) | static_cast<std::uint32_t>(bit);
        bit_hist_ = (bit_hist_ << 1) | static_cast<std::uint32_t>(bit);
    }

    void finish_byte(std::uint8_t byte) {
        cm_low_.update_byte(byte);
        cm_high_.update_byte(byte);
        word_.update(byte, wrt_mode_);
        rec_.update(pos_ + 1u, byte, c1_);
        ind_.update(byte, c1_, c2_);
        if (exe_mode_) exe_.update(byte);
        if (ppm_) ppm_->update_byte(pos_, byte);
        if (lstm_) lstm_->update_byte(byte);
    }

  private:
    const std::uint8_t* buf_;
    CmConfig cfg_;
    bool deep_;
    bool exe_mode_;
    bool wrt_mode_;
    bool use_ppm_;
    bool use_lstm_;
    MemBudget budget_;

    ContextMap cm_low_;
    ContextMap cm_high_;
    DirectModels direct_;
    MatchModel match_;
    std::unique_ptr<PpmModel> ppm_;
    std::unique_ptr<LstmModel> lstm_;
    WordTracker word_;
    RecordTracker rec_;
    IndirectTracker ind_;
    ExeTracker exe_;
    FlatProbMap bit_pm_;

    LogisticMixer mix_c1_;
    LogisticMixer mix_c0_;
    LogisticMixer mix_state_;
    LogisticMixer mix_cls_;
    LogisticMixer mix_c12_;
    LogisticMixer meta_mix_;

    APM apm1_;
    APM apm2_;
    APM apm3_;
    APM apm4_;

    std::array<std::int16_t, kMixerInputs> inputs_{};
    std::array<std::int16_t, 16> meta_inputs_{};

    std::size_t pos_ = 0;
    std::uint8_t c1_ = 0, c2_ = 0, c3_ = 0, c4_ = 0, c5_ = 0;
    std::uint32_t c1_cls_ = 0, c2_cls_ = 0;
    std::uint32_t c0_ = 1u;
    std::uint32_t sub_ = 1u;
    std::uint32_t bit_hist_ = 0u;
    std::uint32_t last_err_ = 0u;
    std::uint32_t last_p_final_ = 32768u;
    std::size_t last_bit_idx_ = 0u;
};

}  // namespace

// ---------------------------------------------------------------------------
// Public FZAC v3 API
// ---------------------------------------------------------------------------
auto choose_cm_config(std::span<const std::byte> data, int level) -> CmConfig {
    CmConfig cfg{};
    const auto* raw = reinterpret_cast<const std::uint8_t*>(data.data());
    const std::size_t n = data.size();

    std::uint8_t ml = 0;
    if (n <= (64u << 10))       ml = 0;
    else if (n <= (256u << 10)) ml = 1;
    else if (n <= (1u << 20))   ml = 2;
    else if (n <= (4u << 20))   ml = 3;
    else if (n <= (16u << 20))  ml = 4;
    else if (n <= (64u << 20))  ml = 5;
    else if (n <= (256u << 20)) ml = 6;
    else if (n <= (512u << 20)) ml = 7;
    else                        ml = 8;

    if (level <= 7 && ml > 4u) ml = 4u;
    else if (level == 8 && ml > 5u) ml = 5u;
    else if (level == 9 && ml > 7u) ml = 7u;
    cfg.mem_level = ml;

    if (level >= 8) {
        cfg.flags |= kCmPPM;
    }
    if (level >= 9) {
        cfg.flags |= kCmDeepOrders;
    }
    if (level >= 10 && n >= 4096u) {
        cfg.flags |= kCmLSTM;
        // extra bits 1..2: 1 -> H=64 (L10-11), 2 -> H=96 (L12)
        const std::uint8_t lstm_tier = (level >= 12) ? 2u : 1u;
        cfg.extra |= static_cast<std::uint8_t>(lstm_tier << 1);
    }

    bool use_jcc = false;
    if (detect_bcj(raw, n, use_jcc)) {
        cfg.transform = kTransformBcjX86;
        if (use_jcc) cfg.extra |= 1u;
        cfg.flags |= kCmExe;
    } else {
        const std::size_t sample = std::min<std::size_t>(n, 16384u);
        std::size_t printable = 0u;
        for (std::size_t i = 0; i < sample; ++i) {
            const std::uint8_t b = raw[i];
            if ((b >= 32u && b <= 126u) || b == '\n' || b == '\r' || b == '\t') {
                ++printable;
            }
        }
        if (sample > 0u && printable * 100u >= sample * 85u) {
            cfg.flags |= kCmTextMode;
        }
    }
    return cfg;
}

auto cm_encode_raw(const std::uint8_t* src, std::size_t n, const CmConfig& cfg)
    -> std::vector<std::byte> {
    RangeEncoder16 rc;
    rc.reserve(n / 3u + 64u);
    auto pred = std::make_unique<CmPredictor>(src, cfg);
    for (std::size_t pos = 0; pos < n; ++pos) {
        pred->start_byte(pos);
        const std::uint8_t byte = src[pos];
        for (int bpos = 0; bpos < 8; ++bpos) {
            const std::uint32_t p16 = pred->predict_bit(bpos);
            const int bit = (byte >> (7 - bpos)) & 1;
            rc.encode_bit(p16, bit);
            pred->update_bit(bpos, bit);
        }
        pred->finish_byte(byte);
    }
    rc.finish();
    return std::move(rc.buffer());
}

void cm_decode_raw(const std::byte* in_begin, const std::byte* in_end,
                   std::uint8_t* dst, std::size_t n, const CmConfig& cfg) {
    RangeDecoder16 rd(in_begin, in_end);
    auto pred = std::make_unique<CmPredictor>(dst, cfg);
    for (std::size_t pos = 0; pos < n; ++pos) {
        pred->start_byte(pos);
        std::uint8_t byte = 0u;
        for (int bpos = 0; bpos < 8; ++bpos) {
            const std::uint32_t p16 = pred->predict_bit(bpos);
            const int bit = rd.decode_bit(p16);
            byte = static_cast<std::uint8_t>((byte << 1) | bit);
            pred->update_bit(bpos, bit);
        }
        dst[pos] = byte;
        pred->finish_byte(byte);
    }
}

namespace {

auto fzac_v3_compress_single(std::span<const std::byte> data, int level,
                             std::uint8_t max_mem_level) -> std::vector<std::byte> {
    if (data.empty()) return {};
    const auto* raw = reinterpret_cast<const std::uint8_t*>(data.data());
    const std::size_t n = data.size();

    CmConfig cfg = choose_cm_config(data, level);
    if (cfg.mem_level > max_mem_level) cfg.mem_level = max_mem_level;

    std::vector<std::uint8_t> transformed;
    const std::uint8_t* src = raw;
    std::size_t src_len = n;
    std::vector<std::byte> dict_comp;
    std::size_t dict_raw_len = 0u;

    if (cfg.transform == kTransformBcjX86) {
        transformed.assign(raw, raw + n);
        bcj_e8e9_forward(transformed.data(), n);
        if ((cfg.extra & 1u) != 0u) {
            bcj_jcc_forward(transformed.data(), n);
        }
        src = transformed.data();
    } else if (cfg.transform == kTransformRaw && (cfg.flags & kCmTextMode) != 0u &&
               n >= (4u << 20)) {
        auto wrt = wrt_forward(raw, n);
        if (wrt.beneficial) {
            cfg.transform = kTransformWrt;
            dict_raw_len = wrt.dict_blob.size();
            CmConfig dict_cfg{};
            dict_cfg.mem_level = (dict_raw_len > (128u << 10)) ? 1u : 0u;
            dict_cfg.flags = kCmTextMode;
            dict_cfg.transform = kTransformRaw;
            dict_comp = cm_encode_raw(
                reinterpret_cast<const std::uint8_t*>(wrt.dict_blob.data()),
                dict_raw_len, dict_cfg);
            transformed = std::move(wrt.stream);
            src = transformed.data();
            src_len = transformed.size();
        }
    }

    auto payload = cm_encode_raw(src, src_len, cfg);

    std::vector<std::byte> out;
    out.reserve(payload.size() + dict_comp.size() + 32u);
    out.push_back(static_cast<std::byte>(0x46u));
    out.push_back(static_cast<std::byte>(0x5Au));
    const std::uint8_t hdr_flags = static_cast<std::uint8_t>(
        3u | ((cfg.transform & 0x03u) << 2) | ((cfg.mem_level & 0x0Fu) << 4));
    out.push_back(static_cast<std::byte>(hdr_flags));
    append_varint(out, n);
    out.push_back(static_cast<std::byte>(cfg.flags));
    out.push_back(static_cast<std::byte>(cfg.extra));
    if (cfg.transform == kTransformWrt) {
        append_varint(out, dict_raw_len);
        append_varint(out, dict_comp.size());
        append_varint(out, src_len);
        out.insert(out.end(), dict_comp.begin(), dict_comp.end());
    }
    out.insert(out.end(), payload.begin(), payload.end());

    if (out.size() >= n) return {};
    return out;
}

}  // namespace

auto fzac_v3_compress(std::span<const std::byte> data, int level)
    -> std::vector<std::byte> {
    if (data.empty()) return {};
    const std::size_t n = data.size();
    const unsigned hw = detail::effective_threads();

    // For huge inputs (> 128 MiB, e.g. 1 GB enwik9), split into up to 4 large
    // parallel blocks (~250 MB each) so 4 workers run concurrently within <= 9 GB
    // total RAM (each worker capped to mem_level 6 ~ 2.2 GB) and finish in ~15-30 min!
    constexpr std::size_t kParallelMinSize = 128u << 20;
    if (hw < 2u || n <= kParallelMinSize) {
        return fzac_v3_compress_single(data, level, /*max_mem_level=*/8u);
    }

    const std::size_t worker_cap = (n > (512u << 20)) ? 5u : 4u;
    const std::size_t max_workers = std::min<std::size_t>(hw, worker_cap);
    const std::size_t block_size = (n + max_workers - 1u) / max_workers;
    const std::size_t num_blocks = (n + block_size - 1u) / block_size;
    // Memory cap per worker so total concurrent RSS stays <= ~8.5 GB (< 10 GB budget)
    const std::uint8_t per_worker_ml = (num_blocks >= 4u) ? 5u : 6u;

    std::vector<std::vector<std::byte>> blk_out(num_blocks);
    std::vector<std::size_t> blk_raw(num_blocks, 0u);
    std::atomic<std::size_t> next_blk{0u};
    std::vector<std::thread> workers;
    workers.reserve(num_blocks);
    for (std::size_t t = 0; t < num_blocks; ++t) {
        workers.emplace_back([&]() {
            for (;;) {
                const std::size_t b = next_blk.fetch_add(1u, std::memory_order_relaxed);
                if (b >= num_blocks) break;
                const std::size_t off = b * block_size;
                const std::size_t blen = std::min(block_size, n - off);
                blk_raw[b] = blen;
                blk_out[b] = fzac_v3_compress_single(
                    data.subspan(off, blen), level, per_worker_ml);
            }
        });
    }
    for (auto& w : workers) w.join();

    std::vector<std::byte> out;
    out.push_back(static_cast<std::byte>(0x46u));
    out.push_back(static_cast<std::byte>(0x5Au));
    const std::uint8_t hdr_flags = static_cast<std::uint8_t>(
        3u | (kTransformMultiBlock << 2));
    out.push_back(static_cast<std::byte>(hdr_flags));
    append_varint(out, n);
    append_varint(out, num_blocks);

    std::size_t payload_total = 0u;
    for (std::size_t b = 0; b < num_blocks; ++b) {
        append_varint(out, blk_raw[b]);
        if (blk_out[b].empty()) {
            append_varint(out, 0u);
            payload_total += blk_raw[b];
        } else {
            append_varint(out, blk_out[b].size());
            payload_total += blk_out[b].size();
        }
    }
    if (out.size() + payload_total >= n) return {};

    out.reserve(out.size() + payload_total);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        if (blk_out[b].empty()) {
            const std::size_t off = b * block_size;
            out.insert(out.end(), data.data() + off, data.data() + off + blk_raw[b]);
        } else {
            out.insert(out.end(), blk_out[b].begin(), blk_out[b].end());
        }
    }
    return out;
}

void fzac_v3_decompress(const std::byte* payload_begin,
                        const std::byte* payload_end,
                        std::uint8_t hdr_flags,
                        std::uint8_t* dst,
                        std::size_t raw_size) {
    if (raw_size == 0u) return;
    const std::size_t avail = static_cast<std::size_t>(payload_end - payload_begin);
    const auto* in = reinterpret_cast<const std::uint8_t*>(payload_begin);
    const std::uint8_t transform = (hdr_flags >> 2) & 0x03u;

    if (transform == kTransformMultiBlock) {
        std::size_t off = 0u;
        const std::uint64_t nb64 =
            read_varint(in, avail, off, "truncated FZAC v3 multi-block count");
        if (nb64 < 2u || nb64 > raw_size) {
            throw std::runtime_error("invalid FZAC v3 multi-block count");
        }
        const std::size_t num_blocks = static_cast<std::size_t>(nb64);
        struct BlkDesc {
            std::size_t in_off = 0;
            std::size_t comp_len = 0;
            std::size_t out_off = 0;
            std::size_t raw_len = 0;
            bool stored = false;
        };
        std::vector<BlkDesc> blks(num_blocks);
        std::size_t sum_raw = 0u;
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const std::uint64_t br =
                read_varint(in, avail, off, "truncated FZAC v3 block directory");
            const std::uint64_t bp =
                read_varint(in, avail, off, "truncated FZAC v3 block directory");
            if (br == 0u || br > raw_size - sum_raw) {
                throw std::runtime_error("invalid FZAC v3 block raw length");
            }
            blks[b].out_off = sum_raw;
            blks[b].raw_len = static_cast<std::size_t>(br);
            blks[b].stored = (bp == 0u);
            blks[b].comp_len = static_cast<std::size_t>(bp);
            sum_raw += blks[b].raw_len;
        }
        if (sum_raw != raw_size) {
            throw std::runtime_error("FZAC v3 multi-block size mismatch");
        }
        std::size_t cursor = off;
        for (auto& blk : blks) {
            const std::size_t need = blk.stored ? blk.raw_len : blk.comp_len;
            if (need > avail - cursor) {
                throw std::runtime_error("truncated FZAC v3 multi-block payload");
            }
            blk.in_off = cursor;
            cursor += need;
        }

        auto decode_one = [&](std::size_t b) {
            const auto& blk = blks[b];
            if (blk.stored) {
                std::memcpy(dst + blk.out_off, payload_begin + blk.in_off, blk.raw_len);
                return;
            }
            if (blk.comp_len < 4u) {
                throw std::runtime_error("FZAC v3 sub-block too short");
            }
            const std::uint8_t* sub = in + blk.in_off;
            if (sub[0] != 0x46u || sub[1] != 0x5Au || (sub[2] & 0x03u) != 3u) {
                throw std::runtime_error("invalid FZAC v3 sub-block header");
            }
            std::size_t sub_off = 3u;
            const std::uint64_t sub_raw =
                read_varint(sub, blk.comp_len, sub_off, "truncated FZAC v3 sub-block size");
            if (sub_raw != blk.raw_len) {
                throw std::runtime_error("FZAC v3 sub-block size mismatch");
            }
            fzac_v3_decompress(payload_begin + blk.in_off + sub_off,
                               payload_begin + blk.in_off + blk.comp_len,
                               sub[2], dst + blk.out_off, blk.raw_len);
        };

        const unsigned hw = detail::effective_threads();
        const std::size_t nw = std::min<std::size_t>(std::max<unsigned>(1u, hw), std::min<std::size_t>(num_blocks, 5u));
        if (nw <= 1u) {
            for (std::size_t b = 0; b < num_blocks; ++b) decode_one(b);
        } else {
            std::atomic<std::size_t> next_blk{0u};
            std::vector<std::exception_ptr> errs(nw, nullptr);
            std::vector<std::thread> workers;
            workers.reserve(nw);
            for (std::size_t t = 0; t < nw; ++t) {
                workers.emplace_back([&, t]() {
                    try {
                        for (;;) {
                            const std::size_t b = next_blk.fetch_add(1u, std::memory_order_relaxed);
                            if (b >= num_blocks) break;
                            decode_one(b);
                        }
                    } catch (...) {
                        errs[t] = std::current_exception();
                    }
                });
            }
            for (auto& w : workers) w.join();
            for (const auto& e : errs) {
                if (e) std::rethrow_exception(e);
            }
        }
        return;
    }

    if (avail < 2u) {
        throw std::runtime_error("truncated FZAC v3 header");
    }

    CmConfig cfg{};
    cfg.transform = transform;
    cfg.mem_level = (hdr_flags >> 4) & 0x0Fu;
    cfg.flags = in[0];
    cfg.extra = in[1];

    if (cfg.transform == kTransformWrt) {
        std::size_t off = 2u;
        const std::uint64_t dict_raw_len64 =
            read_varint(in, avail, off, "truncated FZAC v3 WRT dict raw size");
        const std::uint64_t dict_comp_len64 =
            read_varint(in, avail, off, "truncated FZAC v3 WRT dict comp size");
        const std::uint64_t stream_len64 =
            read_varint(in, avail, off, "truncated FZAC v3 WRT stream size");
        if (dict_comp_len64 > avail - off) {
            throw std::runtime_error("truncated FZAC v3 WRT dictionary payload");
        }
        const std::size_t dict_raw_len = static_cast<std::size_t>(dict_raw_len64);
        const std::size_t dict_comp_len = static_cast<std::size_t>(dict_comp_len64);
        const std::size_t stream_len = static_cast<std::size_t>(stream_len64);

        std::vector<std::uint8_t> dict_blob(dict_raw_len);
        CmConfig dict_cfg{};
        dict_cfg.mem_level = (dict_raw_len > (128u << 10)) ? 1u : 0u;
        dict_cfg.flags = kCmTextMode;
        dict_cfg.transform = kTransformRaw;
        cm_decode_raw(payload_begin + off, payload_begin + off + dict_comp_len,
                      dict_blob.data(), dict_raw_len, dict_cfg);
        off += dict_comp_len;

        std::vector<std::uint8_t> wrt_stream(stream_len);
        cm_decode_raw(payload_begin + off, payload_end,
                      wrt_stream.data(), stream_len, cfg);

        wrt_inverse(dict_blob.data(), dict_raw_len,
                    wrt_stream.data(), stream_len, dst, raw_size);
        return;
    }

    if (cfg.transform != kTransformRaw && cfg.transform != kTransformBcjX86) {
        throw std::runtime_error("unsupported FZAC v3 transform");
    }

    cm_decode_raw(payload_begin + 2u, payload_end, dst, raw_size, cfg);

    if (cfg.transform == kTransformBcjX86) {
        if ((cfg.extra & 1u) != 0u) {
            bcj_jcc_inverse(dst, raw_size);
        }
        bcj_e8e9_inverse(dst, raw_size);
    }
}

}  // namespace fzip::fzac_v3
