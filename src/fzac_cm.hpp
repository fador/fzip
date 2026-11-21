// fzip — FZAC v3 State-of-the-Art Context Mixing (CM) engine.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fzip::fzac_v3 {

// Feature flags stored in the FZAC v3 stream header so the decoder always
// instantiates the exact same model ensemble and hash table sizes.
enum CmFlags : std::uint8_t {
    kCmNone        = 0x00u,
    kCmPPM         = 0x01u,  // Variable-order byte-level PPMd predictor
    kCmLSTM        = 0x02u,  // Online byte-level neural (LSTM) predictor
    kCmExe         = 0x04u,  // x86/x64 instruction & executable context model
    kCmDeepOrders  = 0x08u,  // Extra high orders (O10..O16) & deeper MatchModel
    kCmTextMode    = 0x10u,  // Enhanced text/XML/word/tag models
    kCmDeltaNet    = 0x20u,  // Online DeltaNet associative matrix-memory RNN
    kCmNeuralMix   = 0x40u,  // Residual nonlinear MLP meta-mixer + neural APM
    kCmTransformer = 0x80u,  // Compact online causal Transformer predictor
};

enum TransformId : std::uint8_t {
    kTransformRaw        = 0u,
    kTransformBcjX86     = 1u,  // Bijective x86/x64 E8/E9 (+ optional Jcc) filter
    kTransformWrt        = 2u,  // Dynamic universal Word/Token Replacement Transform
    kTransformMultiBlock = 3u,  // Parallel large-block container (<= 10 GB total RAM)
};

struct CmConfig {
    std::uint8_t mem_level = 2;    // 0..8 (scales hash & match memory up to ~8 GB)
    std::uint8_t flags = 0;        // bitmask of CmFlags
    std::uint8_t transform = 0;    // TransformId
    std::uint8_t extra = 0;        // transform sub-flags (e.g. BCJ Jcc enabled)
};

// Choose the optimal FZAC v3 configuration for `data` at `level` (7..12).
auto choose_cm_config(std::span<const std::byte> data, int level) -> CmConfig;

// Compress `data` using FZAC v3 (returns full stream starting with 'F','Z',flags...).
// Returns an empty vector if the compressed stream is not smaller than `data`.
auto fzac_v3_compress(std::span<const std::byte> data, int level)
    -> std::vector<std::byte>;

// Decompress an FZAC v3 payload (after the 3-byte magic+flags and raw_size varint).
// `transform_id` comes from bits 2..3 of the stream flags byte.
void fzac_v3_decompress(const std::byte* payload_begin,
                        const std::byte* payload_end,
                        std::uint8_t transform_id,
                        std::uint8_t* dst,
                        std::size_t raw_size);

// Raw CM block encode/decode (used both directly and for compressed WRT streams).
auto cm_encode_raw(const std::uint8_t* src, std::size_t n, const CmConfig& cfg)
    -> std::vector<std::byte>;
void cm_decode_raw(const std::byte* in_begin, const std::byte* in_end,
                   std::uint8_t* dst, std::size_t n, const CmConfig& cfg);

}  // namespace fzip::fzac_v3
