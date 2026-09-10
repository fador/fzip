// fzip — zstd compressor implementation (RFC 8878).
//
// Emits spec-compliant compressed blocks: a raw literals section followed by
// a sequences section encoded with the predefined FSE tables (symbol
// compression mode 0). LZ77 uses a greedy hash-chain match finder over a
// 32 KiB window, matching the frame's declared window.
#include "zstd.hpp"
#include "zstd_internal.hpp"

#include <algorithm>
#include <cstring>

#include "xxhash.hpp"
#include "zstd_fse.hpp"
#include "zstd_huffman.hpp"
#include "zstd_predefined.hpp"

namespace fzip::zstd {

namespace {

// --- LZ77 match finder (3-byte hash, hash-chain, 32 KiB window) ---
constexpr int kHashBits = 16;
constexpr int kHashSize = 1 << kHashBits;
constexpr int kHashMask = kHashSize - 1;
constexpr int kWindow = 131072;   // 128 KiB
constexpr int kMinMatch = 3;
constexpr int kMaxMatch = 131074;  // zstd max match length

// Block content (and decompressed size) is capped by min(window, 128 KiB).
constexpr int kBlockSize = 131072;

struct Match {
    int distance = 0;
    int length = 0;
};

inline auto hash3(const std::uint8_t* p) -> std::uint32_t {
    std::uint32_t h = (static_cast<std::uint32_t>(p[0]) << 16) |
                      (static_cast<std::uint32_t>(p[1]) << 8) |
                      static_cast<std::uint32_t>(p[2]);
    return (h * 2654435761u) >> (32 - kHashBits);
}

auto find_match(const std::uint8_t* data, std::size_t size, std::size_t pos,
                const std::vector<int>& head, const std::vector<int>& prev,
                int effort) -> Match {
    Match best;
    if (pos + kMinMatch > size) return best;
    std::uint32_t h = hash3(data + pos);
    int cand = head[h];
    int limit = static_cast<int>(pos) - kWindow;
    if (limit < 0) limit = 0;
    int tries = effort;
    while (cand >= 0 && tries-- > 0) {
        if (cand < limit) break;
        if (best.length > kMinMatch &&
            data[cand + best.length - 1] != data[pos + best.length - 1]) {
            cand = prev[static_cast<std::size_t>(cand) & (kWindow - 1)];
            continue;
        }
        int maxl = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(kMaxMatch), size - pos));
        int l = 0;
        while (l < maxl && data[cand + l] == data[pos + l]) ++l;
        if (l >= kMinMatch && l > best.length) {
            best.length = l;
            best.distance = static_cast<int>(pos) - cand;
            if (l >= kMaxMatch) break;
        }
        cand = prev[static_cast<std::size_t>(cand) & (kWindow - 1)];
    }
    return best;
}

void insert_hash(const std::uint8_t* data, std::size_t size, std::size_t pos,
                 std::vector<int>& head, std::vector<int>& prev) {
    if (pos + kMinMatch > size) return;
    std::uint32_t h = hash3(data + pos);
    int p = static_cast<int>(pos);
    prev[static_cast<std::size_t>(p) & (kWindow - 1)] = head[h];
    head[h] = p;
}

// One sequence: `lit_len` literals, then a match of `match_len` at
// `match_dist` bytes back.
struct Seq {
    int lit_len = 0;
    int match_len = 0;
    int match_dist = 0;
};

auto build_sequences(const std::uint8_t* data, std::size_t size, int effort,
                     std::vector<std::uint8_t>& literals) -> std::vector<Seq> {
    std::vector<Seq> seqs;
    literals.clear();

    std::vector<int> head(kHashSize, -1);
    std::vector<int> prev(kWindow, -1);

    std::size_t pos = 0;
    std::size_t lit_start = 0;
    while (pos < size) {
        Match m = find_match(data, size, pos, head, prev, effort);
        if (m.length >= kMinMatch) {
            for (std::size_t i = lit_start; i < pos; ++i) {
                literals.push_back(data[i]);
            }
            Seq s;
            s.lit_len = static_cast<int>(pos - lit_start);
            s.match_len = m.length;
            s.match_dist = m.distance;
            seqs.push_back(s);
            for (int i = 0; i < m.length; ++i) {
                insert_hash(data, size, pos + i, head, prev);
            }
            pos += static_cast<std::size_t>(m.length);
            lit_start = pos;
        } else {
            insert_hash(data, size, pos, head, prev);
            ++pos;
        }
    }
    for (std::size_t i = lit_start; i < size; ++i) {
        literals.push_back(data[i]);
    }
    return seqs;
}

// --- Code tables (RFC 8878 §4.2.2) ---
struct Code {
    int code = 0;
    int extra_bits = 0;
    int extra_val = 0;
};

Code lit_len_code(int len) {
    static const int kBase[36] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048,
        4096, 8192, 16384, 32768, 65536};
    static const int kExtra[36] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    for (int c = 35; c >= 0; --c) {
        if (len >= kBase[c]) return {c, kExtra[c], len - kBase[c]};
    }
    return {0, 0, 0};
}

Code match_len_code(int len) {
    static const int kBase[53] = {
        3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027,
        2051, 4099, 8195, 16387, 32771, 65539};
    static const int kExtra[53] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    for (int c = 52; c >= 0; --c) {
        if (len >= kBase[c]) return {c, kExtra[c], len - kBase[c]};
    }
    return {0, 0, 0};
}

// Offset code: Offset_Value = (1 << code) + extra, offset = Offset_Value - 3.
// Codes 0..1 are repeat codes; explicit offsets use code >= 2.
Code offset_code(int dist) {
    int v = dist + 3;
    int code = 0;
    while ((1 << (code + 1)) <= v) ++code;
    if (code < 2) code = 2;  // never use repeat codes
    return {code, code, v - (1 << code)};
}

// --- Predefined normalized distributions (RFC 8878 §4.1.1) ---
constexpr std::int16_t kLLNorm[36] = {
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1};
constexpr std::int16_t kOFNorm[29] = {
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1};
constexpr std::int16_t kMLNorm[53] = {
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
    -1, -1, -1, -1, -1};

const FseCTable& ll_ctable() {
    static const FseCTable ct = build_fse_ctable(6, kLLNorm, 35);
    return ct;
}
const FseCTable& of_ctable() {
    static const FseCTable ct = build_fse_ctable(5, kOFNorm, 28);
    return ct;
}
const FseCTable& ml_ctable() {
    static const FseCTable ct = build_fse_ctable(6, kMLNorm, 52);
    return ct;
}

// --- Section emission ---
// Write a little-endian bitfield header for the literals section.
void put_literals_header(std::vector<std::byte>& out, int type, int size_format,
                         int regen, int csize, int k) {
    std::uint64_t v = static_cast<std::uint64_t>(type) |
                      (static_cast<std::uint64_t>(size_format) << 2) |
                      (static_cast<std::uint64_t>(regen) << 4) |
                      (static_cast<std::uint64_t>(csize) << (4 + k));
    const int nbytes = (4 + 2 * k) / 8;
    for (int i = 0; i < nbytes; ++i) {
        out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
    }
}

// Emit the literals section, choosing Huffman (Compressed), RLE, or Raw.
void emit_literals(std::vector<std::byte>& out,
                   const std::vector<std::uint8_t>& lits) {
    const int n = static_cast<int>(lits.size());

    // Try Huffman-coded literals (4-stream).
    if (n >= 64) {
        std::vector<std::byte> body;
        if (huf_compress_literals(lits.data(), n, body)) {
            const int m = std::max(n, static_cast<int>(body.size()));
            int size_format;
            int k;
            if (m <= 1023) {
                size_format = 1; k = 10;
            } else if (m <= 16383) {
                size_format = 2; k = 14;
            } else {
                size_format = 3; k = 18;
            }
            const int hdr_bytes = (4 + 2 * k) / 8;
            if (static_cast<int>(body.size()) + hdr_bytes < n) {
                put_literals_header(out, 2, size_format, n,
                                    static_cast<int>(body.size()), k);
                out.insert(out.end(), body.begin(), body.end());
                return;
            }
        }
    }

    // RLE literals.
    bool all_same = n > 0;
    for (int i = 1; i < n; ++i) {
        if (lits[static_cast<std::size_t>(i)] != lits[0]) {
            all_same = false;
            break;
        }
    }
    if (all_same) {
        if (n <= 31) {
            out.push_back(static_cast<std::byte>((n << 3) | (0 << 2) | 1));
        } else if (n <= 4095) {
            out.push_back(static_cast<std::byte>(((n & 0xF) << 4) | (1 << 2) | 1));
            out.push_back(static_cast<std::byte>((n >> 4) & 0xFF));
        } else {
            out.push_back(static_cast<std::byte>(((n & 0xF) << 4) | (3 << 2) | 1));
            out.push_back(static_cast<std::byte>((n >> 4) & 0xFF));
            out.push_back(static_cast<std::byte>((n >> 12) & 0xFF));
        }
        out.push_back(static_cast<std::byte>(lits[0]));
        return;
    }

    // Raw literals.
    if (n <= 31) {
        // 1-byte header, Size_Format 0, type 0 (Raw).
        out.push_back(static_cast<std::byte>((n << 3)));
    } else if (n <= 4095) {
        // 2-byte header, Size_Format 1.
        out.push_back(static_cast<std::byte>(((n & 0xF) << 4) | (1 << 2)));
        out.push_back(static_cast<std::byte>((n >> 4) & 0xFF));
    } else {
        // 3-byte header, Size_Format 3.
        out.push_back(static_cast<std::byte>(((n & 0xF) << 4) | (3 << 2)));
        out.push_back(static_cast<std::byte>((n >> 4) & 0xFF));
        out.push_back(static_cast<std::byte>((n >> 12) & 0xFF));
    }
    for (std::uint8_t b : lits) out.push_back(static_cast<std::byte>(b));
}

// A per-stream sequence FSE table and its symbol-compression mode.
struct SeqStream {
    int mode = 0;  // 0 predefined, 1 RLE, 2 FSE
    int rle_sym = 0;
    FseCTable ct;
    std::vector<std::byte> desc;  // NCount table description (mode 2)
};

auto make_seq_stream(const int* freq, int max_code, int max_log, int n,
                     const FseCTable& predef) -> SeqStream {
    SeqStream s;
    int present = 0;
    int only = 0;
    for (int c = 0; c <= max_code; ++c) {
        if (freq[c] > 0) {
            ++present;
            only = c;
        }
    }
    if (present <= 1) {
        s.mode = 1;  // RLE
        s.rle_sym = only;
        return s;
    }
    if (n < 32) {
        s.mode = 0;  // predefined (table description would cost more)
        s.ct = predef;
        return s;
    }
    s.mode = 2;  // inline FSE table
    std::vector<unsigned> counts(static_cast<std::size_t>(max_code) + 1);
    for (int c = 0; c <= max_code; ++c) {
        counts[static_cast<std::size_t>(c)] =
            static_cast<unsigned>(freq[c]);
    }
    std::vector<int> norm;
    fse_normalize(counts.data(), n, max_code, max_log, norm);
    std::vector<std::int16_t> norm16(static_cast<std::size_t>(max_code) + 1);
    for (int c = 0; c <= max_code; ++c) {
        norm16[static_cast<std::size_t>(c)] =
            static_cast<std::int16_t>(norm[static_cast<std::size_t>(c)]);
    }
    s.ct = build_fse_ctable(max_log, norm16.data(), max_code);
    fse_write_ncount(s.desc, norm.data(), max_code, max_log);
    return s;
}

void emit_seq_table(std::vector<std::byte>& out, const SeqStream& s) {
    if (s.mode == 1) {
        out.push_back(static_cast<std::byte>(s.rle_sym));
    } else if (s.mode == 2) {
        out.insert(out.end(), s.desc.begin(), s.desc.end());
    }
}

void emit_sequences(std::vector<std::byte>& out, const std::vector<Seq>& seqs) {
    const int n = static_cast<int>(seqs.size());
    if (n == 0) {
        out.push_back(std::byte{0});
        return;
    }
    // Number_of_Sequences.
    if (n < 128) {
        out.push_back(static_cast<std::byte>(n));
    } else if (n <= 0x7EFF) {
        out.push_back(static_cast<std::byte>(128 + (n >> 8)));
        out.push_back(static_cast<std::byte>(n & 0xFF));
    } else {
        int v = n - 0x7F00;
        out.push_back(static_cast<std::byte>(255));
        out.push_back(static_cast<std::byte>(v & 0xFF));
        out.push_back(static_cast<std::byte>((v >> 8) & 0xFF));
    }

    // Precompute codes and per-block frequencies.
    std::vector<int> llc(static_cast<std::size_t>(n));
    std::vector<int> ofc(static_cast<std::size_t>(n));
    std::vector<int> mlc(static_cast<std::size_t>(n));
    int ll_freq[36] = {0};
    int of_freq[32] = {0};
    int ml_freq[53] = {0};
    for (int i = 0; i < n; ++i) {
        const Seq& s = seqs[static_cast<std::size_t>(i)];
        llc[static_cast<std::size_t>(i)] = lit_len_code(s.lit_len).code;
        ofc[static_cast<std::size_t>(i)] = offset_code(s.match_dist).code;
        mlc[static_cast<std::size_t>(i)] = match_len_code(s.match_len).code;
        ll_freq[llc[static_cast<std::size_t>(i)]]++;
        of_freq[ofc[static_cast<std::size_t>(i)]]++;
        ml_freq[mlc[static_cast<std::size_t>(i)]]++;
    }
    SeqStream ll = make_seq_stream(ll_freq, 35, 9, n, ll_ctable());
    SeqStream of = make_seq_stream(of_freq, 31, 8, n, of_ctable());
    SeqStream ml = make_seq_stream(ml_freq, 52, 9, n, ml_ctable());

    // Symbol compression modes + table descriptions (LL, OF, ML order).
    out.push_back(static_cast<std::byte>((ll.mode << 6) | (of.mode << 4) |
                                         (ml.mode << 2)));
    emit_seq_table(out, ll);
    emit_seq_table(out, of);
    emit_seq_table(out, ml);

    FseBitWriter w;
    std::uint32_t ll_state = 0, of_state = 0, ml_state = 0;

    // Last sequence: its FSE symbols live in the initial states; only its
    // extra bits are written here.
    {
        const Seq& s = seqs[static_cast<std::size_t>(n - 1)];
        Code lc = lit_len_code(s.lit_len);
        Code mc = match_len_code(s.match_len);
        Code oc = offset_code(s.match_dist);
        if (ll.mode != 1) ll_state = fse_init_cstate2(ll.ct, lc.code);
        if (of.mode != 1) of_state = fse_init_cstate2(of.ct, oc.code);
        if (ml.mode != 1) ml_state = fse_init_cstate2(ml.ct, mc.code);
        if (lc.extra_bits) w.put_bits(static_cast<std::uint32_t>(lc.extra_val), lc.extra_bits);
        if (mc.extra_bits) w.put_bits(static_cast<std::uint32_t>(mc.extra_val), mc.extra_bits);
        if (oc.extra_bits) w.put_bits(static_cast<std::uint32_t>(oc.extra_val), oc.extra_bits);
    }
    for (int i = n - 2; i >= 0; --i) {
        const Seq& s = seqs[static_cast<std::size_t>(i)];
        Code lc = lit_len_code(s.lit_len);
        Code mc = match_len_code(s.match_len);
        Code oc = offset_code(s.match_dist);
        if (of.mode != 1) fse_encode_symbol(w, of.ct, of_state, oc.code);
        if (ml.mode != 1) fse_encode_symbol(w, ml.ct, ml_state, mc.code);
        if (ll.mode != 1) fse_encode_symbol(w, ll.ct, ll_state, lc.code);
        if (lc.extra_bits) w.put_bits(static_cast<std::uint32_t>(lc.extra_val), lc.extra_bits);
        if (mc.extra_bits) w.put_bits(static_cast<std::uint32_t>(mc.extra_val), mc.extra_bits);
        if (oc.extra_bits) w.put_bits(static_cast<std::uint32_t>(oc.extra_val), oc.extra_bits);
    }
    if (ml.mode != 1) fse_flush_cstate(w, ml.ct, ml_state);
    if (of.mode != 1) fse_flush_cstate(w, of.ct, of_state);
    if (ll.mode != 1) fse_flush_cstate(w, ll.ct, ll_state);
    w.put_bit(true);
    w.align_to_byte();
    const auto& bytes = w.data();
    out.insert(out.end(), bytes.begin(), bytes.end());
}

// Build the content of a compressed block. Returns false if the block cannot
// be represented as a compressed block (should fall back to a raw block).
auto build_compressed_block(const std::uint8_t* data, std::size_t size,
                            int effort, std::vector<std::byte>& out) -> bool {
    std::vector<std::uint8_t> literals;
    std::vector<Seq> seqs = build_sequences(data, size, effort, literals);
    if (seqs.empty()) return false;  // nothing to gain from a compressed block
    out.clear();
    emit_literals(out, literals);
    emit_sequences(out, seqs);
    return out.size() < size;
}

}  // namespace

auto compress(std::span<const std::byte> data, int level,
              int /*long_distance_log*/) -> std::vector<std::byte> {
    if (data.empty()) return {};
    if (level < 1) level = 1;
    if (level > 22) level = 22;
    // Map zstd level to a match-finder effort.
    int effort = 1 << std::min(level, 12);

    std::size_t size = data.size();

    std::vector<std::byte> output;
    std::uint32_t magic = 0xFD2FB528u;
    output.insert(output.end(), reinterpret_cast<std::byte*>(&magic),
                  reinterpret_cast<std::byte*>(&magic) + 4);

    int fcs_size;
    std::uint8_t fcs_code;
    std::uint64_t fcs_value;
    if (size < 256) {
        fcs_code = 0; fcs_size = 1; fcs_value = size;
    } else if (size < 256 + 65536) {
        // A 2-byte Frame_Content_Size stores the value minus 256.
        fcs_code = 1; fcs_size = 2; fcs_value = size - 256;
    } else if (size < (1ULL << 32)) {
        fcs_code = 2; fcs_size = 4; fcs_value = size;
    } else {
        fcs_code = 3; fcs_size = 8; fcs_value = size;
    }
    bool single_seg = (fcs_code == 0);
    std::uint8_t desc = static_cast<std::uint8_t>(
        (1 << 2) | (single_seg ? (1 << 5) : 0) | (fcs_code << 6));
    output.push_back(static_cast<std::byte>(desc));
    if (!single_seg) {
        output.push_back(static_cast<std::byte>(56));  // window = 128 KiB
    }
    for (int i = 0; i < fcs_size; ++i) {
        output.push_back(static_cast<std::byte>((fcs_value >> (8 * i)) & 0xFF));
    }

    const auto* p = reinterpret_cast<const std::uint8_t*>(data.data());
    std::size_t off = 0;
    while (off < size) {        std::size_t blen = std::min<std::size_t>(kBlockSize, size - off);
        bool last = (off + blen == size);

        std::vector<std::byte> content;
        bool compressed = build_compressed_block(p + off, blen, effort, content);
        const std::vector<std::byte>* payload;
        std::vector<std::byte> raw;
        if (compressed) {
            payload = &content;
        } else {
            raw.assign(reinterpret_cast<const std::byte*>(p + off),
                       reinterpret_cast<const std::byte*>(p + off) + blen);
            payload = &raw;
        }

        std::uint32_t type = compressed ? 2u : 0u;
        std::uint32_t cs = static_cast<std::uint32_t>(payload->size());
        std::uint32_t blk_hdr = (last ? 1u : 0u) | (type << 1) | (cs << 3);
        output.push_back(static_cast<std::byte>(blk_hdr & 0xFF));
        output.push_back(static_cast<std::byte>((blk_hdr >> 8) & 0xFF));
        output.push_back(static_cast<std::byte>((blk_hdr >> 16) & 0xFF));
        output.insert(output.end(), payload->begin(), payload->end());
        off += blen;
    }

    std::uint64_t checksum = xxhash64(data);
    std::uint32_t csum = static_cast<std::uint32_t>(checksum);
    output.insert(output.end(), reinterpret_cast<std::byte*>(&csum),
                  reinterpret_cast<std::byte*>(&csum) + 4);

    if (output.size() >= size) return {};
    return output;
}

}  // namespace fzip::zstd
