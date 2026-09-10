// fzip — zstd Huffman codec for literals (RFC 8878 §4.2.1).
#include "zstd_huffman.hpp"

#include <algorithm>
#include <cstring>

#include "zstd_fse.hpp"
#include "zstd_internal.hpp"

namespace fzip::zstd {

namespace {

auto highbit_u32(std::uint32_t x) -> int {
    int r = 0;
    while (x > 1) { x >>= 1; ++r; }
    return r;
}

// Length-limited prefix code lengths via package-merge (Larmore-Hirschberg),
// ported from Stephan Brumme's public-domain implementation. `A` must be a
// histogram sorted ascending with no zeros; on return it holds code lengths.
auto package_merge_sorted(int max_length, std::vector<std::uint64_t>& A) -> int {
    int num_codes = static_cast<int>(A.size());
    if (num_codes == 0 || max_length == 0) return 0;
    if (num_codes <= 2) {
        A[0] = 1;
        if (num_codes == 2) A[1] = 1;
        return 1;
    }
    if (max_length > 63) return 0;
    if ((1ULL << max_length) < static_cast<std::uint64_t>(num_codes)) return 0;

    using BitMask = std::uint64_t;
    const int max_buffer = 2 * num_codes;
    std::vector<std::uint64_t> current(static_cast<std::size_t>(max_buffer));
    std::vector<std::uint64_t> previous(static_cast<std::size_t>(max_buffer));
    std::vector<BitMask> is_merged(static_cast<std::size_t>(max_buffer), 0);
    for (int i = 0; i < num_codes; ++i) previous[i] = A[i];
    int num_previous = num_codes;
    const int num_relevant = 2 * num_codes - 2;

    BitMask mask = 1;
    for (int bits = max_length - 1; bits > 0; --bits) {
        num_previous &= ~1;
        current[0] = A[0];
        current[1] = A[1];
        std::uint64_t sum = current[0] + current[1];
        int num_current = 2;
        int num_hist = 2;
        int num_merged = 0;
        for (;;) {
            if (num_hist < num_codes && A[num_hist] <= sum) {
                current[num_current++] = A[num_hist++];
                continue;
            }
            is_merged[num_current] |= mask;
            current[num_current] = sum;
            ++num_current;
            ++num_merged;
            if (num_merged * 2 >= num_previous) break;
            sum = previous[num_merged * 2] + previous[num_merged * 2 + 1];
        }
        while (num_hist < num_codes) current[num_current++] = A[num_hist++];
        mask <<= 1;

        if (num_previous >= num_relevant) {
            bool keep_going = false;
            for (int i = num_relevant - 1; i > 0; --i) {
                if (previous[i] != current[i]) { keep_going = true; break; }
            }
            if (!keep_going) break;
        }
        std::swap(previous, current);
        num_previous = num_current;
    }
    mask >>= 1;

    std::vector<unsigned int> code_lengths(static_cast<std::size_t>(num_codes), 0);
    int num_analyze = num_relevant;
    while (mask != 0) {
        int num_merged = 0;
        code_lengths[0]++;
        code_lengths[1]++;
        int symbol = 2;
        for (int i = symbol; i < num_analyze; ++i) {
            if ((is_merged[i] & mask) == 0) {
                code_lengths[symbol]++;
                ++symbol;
            } else {
                ++num_merged;
            }
        }
        num_analyze = 2 * num_merged;
        mask >>= 1;
    }
    for (int i = 0; i < num_analyze; ++i) code_lengths[i]++;
    for (int i = 0; i < num_codes; ++i) A[i] = code_lengths[static_cast<std::size_t>(i)];
    return static_cast<int>(code_lengths[0]);
}

// zstd canonical code assignment: symbols are ranked by code length (longest
// first), assigned consecutive codes, per HUF_readCTable's valPerRank.
void finalize_codes(HufTable& t) {
    int max_len = 0;
    for (int l : t.lengths) {
        if (l > max_len) max_len = l;
    }
    t.table_log = max_len;
    t.codes.assign(t.lengths.size(), 0);
    if (max_len == 0) return;

    std::vector<int> nb_per_rank(static_cast<std::size_t>(max_len) + 2, 0);
    for (int l : t.lengths) {
        if (l > 0) nb_per_rank[static_cast<std::size_t>(l)]++;
    }
    std::vector<int> val_per_rank(static_cast<std::size_t>(max_len) + 2, 0);
    int min = 0;
    for (int n = max_len; n > 0; --n) {
        val_per_rank[static_cast<std::size_t>(n)] = min;
        min += nb_per_rank[static_cast<std::size_t>(n)];
        min >>= 1;
    }
    for (std::size_t s = 0; s < t.lengths.size(); ++s) {
        int l = t.lengths[s];
        if (l > 0) {
            t.codes[s] = static_cast<std::uint32_t>(
                val_per_rank[static_cast<std::size_t>(l)]++);
        }
    }
}

// Serialize the tree description in direct (nibble) mode.
auto write_weights(const HufTable& t, std::vector<std::byte>& out) -> bool {
    const int table_log = t.table_log;
    std::vector<int> weights(t.lengths.size(), 0);
    int max_sym = -1;
    for (int s = 0; s < static_cast<int>(t.lengths.size()); ++s) {
        if (t.lengths[s] > 0) {
            weights[static_cast<std::size_t>(s)] = table_log + 1 - t.lengths[s];
            max_sym = s;
        }
    }
    if (max_sym < 1) return false;    // need at least 2 symbols
    if (max_sym > 128) return false;  // direct mode can't represent >128
    const int num_weights = max_sym;  // symbols 0..max_sym-1 encoded
    weights[static_cast<std::size_t>(max_sym)] = 0;  // unused low nibble

    out.push_back(static_cast<std::byte>(127 + num_weights));
    for (int n = 0; n < num_weights; n += 2) {
        int hi = weights[static_cast<std::size_t>(n)] & 0xF;
        int lo = weights[static_cast<std::size_t>(n + 1)] & 0xF;
        out.push_back(static_cast<std::byte>((hi << 4) | lo));
    }
    return true;
}

// Encode one Huffman stream (forward-written, MSB-first codes, end flag).
void encode_stream(const HufTable& t, const std::uint8_t* literals,
                   int n, std::vector<std::byte>& out) {
    FseBitWriter w;
    for (int i = n - 1; i >= 0; --i) {
        std::uint8_t lit = literals[i];
        w.put_bits(t.codes[lit], t.lengths[lit]);
    }
    w.put_bit(true);
    w.align_to_byte();
    const auto& bytes = w.data();
    out.insert(out.end(), bytes.begin(), bytes.end());
}

// Decode one Huffman stream into `n` literals.
void decode_stream(const HufTable& t, const std::byte* data, std::size_t size,
                   int n, std::vector<std::byte>& out) {
    const int dt_log = t.table_log;
    if (dt_log <= 0) throw ZstdError("empty Huffman table");
    struct Entry {
        std::uint8_t sym;
        std::uint8_t bits;
    };
    std::vector<Entry> dt(static_cast<std::size_t>(1) << dt_log);
    for (std::size_t s = 0; s < t.lengths.size(); ++s) {
        int l = t.lengths[s];
        if (l == 0) continue;
        std::uint32_t base = t.codes[s] << (dt_log - l);
        std::uint32_t count = 1u << (dt_log - l);
        for (std::uint32_t i = 0; i < count; ++i) {
            dt[base + i].sym = static_cast<std::uint8_t>(s);
            dt[base + i].bits = static_cast<std::uint8_t>(l);
        }
    }
    FseBitReader r(data, size);
    for (int i = 0; i < n; ++i) {
        std::uint32_t idx = r.peek_bits(dt_log);
        out.push_back(static_cast<std::byte>(dt[idx].sym));
        r.skip_bits(dt[idx].bits);
    }
}

// Per-stream literal counts for 4-stream mode.
void stream_sizes(int total, int sizes[4]) {
    int each = (total + 3) / 4;
    sizes[0] = sizes[1] = sizes[2] = each;
    sizes[3] = total - 3 * each;
    if (sizes[3] < 0) sizes[3] = 0;
}

}  // namespace

auto huf_build(const std::vector<std::uint32_t>& freqs, int max_bits)
    -> HufTable {
    HufTable t;
    t.lengths.assign(freqs.size(), 0);
    struct Sym {
        std::uint64_t f;
        int s;
    };
    std::vector<Sym> syms;
    for (int s = 0; s < static_cast<int>(freqs.size()); ++s) {
        if (freqs[static_cast<std::size_t>(s)] > 0) {
            syms.push_back({freqs[static_cast<std::size_t>(s)], s});
        }
    }
    if (syms.size() < 2) return t;
    std::sort(syms.begin(), syms.end(), [](const Sym& a, const Sym& b) {
        if (a.f != b.f) return a.f < b.f;
        return a.s < b.s;
    });
    std::vector<std::uint64_t> hist(syms.size());
    for (std::size_t i = 0; i < syms.size(); ++i) hist[i] = syms[i].f;
    package_merge_sorted(max_bits, hist);
    for (std::size_t i = 0; i < syms.size(); ++i) {
        t.lengths[static_cast<std::size_t>(syms[i].s)] =
            static_cast<int>(hist[i]);
    }
    finalize_codes(t);
    return t;
}

auto huf_compress_literals(const std::uint8_t* literals, int num_literals,
                           std::vector<std::byte>& out) -> bool {
    if (num_literals < 2) return false;
    std::vector<std::uint32_t> freqs(256, 0);
    for (int i = 0; i < num_literals; ++i) freqs[literals[i]]++;
    HufTable t = huf_build(freqs, 11);
    if (t.table_log == 0) return false;

    std::vector<std::byte> weights;
    if (!write_weights(t, weights)) return false;

    int sizes[4];
    stream_sizes(num_literals, sizes);
    std::vector<std::byte> streams[4];
    int offset = 0;
    for (int i = 0; i < 4; ++i) {
        if (sizes[i] > 0) {
            encode_stream(t, literals + offset, sizes[i], streams[i]);
        }
        offset += sizes[i];
    }

    out.clear();
    out.insert(out.end(), weights.begin(), weights.end());
    // Jump table: sizes of streams 1-3 as 2-byte little-endian.
    for (int i = 0; i < 3; ++i) {
        std::uint32_t s = static_cast<std::uint32_t>(streams[i].size());
        out.push_back(static_cast<std::byte>(s & 0xFF));
        out.push_back(static_cast<std::byte>((s >> 8) & 0xFF));
    }
    for (int i = 0; i < 4; ++i) {
        out.insert(out.end(), streams[i].begin(), streams[i].end());
    }
    return true;
}

auto huf_read_weights(const std::byte* data, std::size_t size, HufTable& t,
                      std::size_t& consumed) -> bool {
    if (size < 1) return false;
    std::uint8_t header = static_cast<std::uint8_t>(data[0]);
    if (header < 128) return false;  // FSE-compressed weights unsupported
    const int num_weights = header - 127;
    const int nbytes = (num_weights + 1) / 2;
    if (1 + static_cast<std::size_t>(nbytes) > size) return false;

    std::vector<int> weights(static_cast<std::size_t>(num_weights) + 2, 0);
    for (int n = 0; n < num_weights; n += 2) {
        std::uint8_t b = static_cast<std::uint8_t>(data[1 + n / 2]);
        weights[static_cast<std::size_t>(n)] = b >> 4;
        weights[static_cast<std::size_t>(n) + 1] = b & 0xF;
    }
    int weight_total = 0;
    for (int n = 0; n < num_weights; ++n) {
        if (weights[static_cast<std::size_t>(n)] > 12) return false;
        weight_total += (1 << weights[static_cast<std::size_t>(n)]) >> 1;
    }
    if (weight_total == 0) return false;
    const int table_log = highbit_u32(static_cast<std::uint32_t>(weight_total)) + 1;
    const int rest = (1 << table_log) - weight_total;
    if (rest <= 0 || (1 << highbit_u32(static_cast<std::uint32_t>(rest))) != rest) {
        return false;
    }
    weights[static_cast<std::size_t>(num_weights)] =
        highbit_u32(static_cast<std::uint32_t>(rest)) + 1;

    t.lengths.assign(256, 0);
    for (int n = 0; n <= num_weights && n < 256; ++n) {
        int w = weights[static_cast<std::size_t>(n)];
        t.lengths[static_cast<std::size_t>(n)] =
            w > 0 ? (table_log + 1 - w) : 0;
    }
    finalize_codes(t);
    consumed = 1 + static_cast<std::size_t>(nbytes);
    return true;
}

auto huf_decode_streams(const HufTable& t, const std::byte* streams,
                        std::size_t body_size, int num_literals,
                        int size_format) -> std::vector<std::byte> {
    std::vector<std::byte> out;
    out.reserve(static_cast<std::size_t>(num_literals));
    if (num_literals == 0) return out;

    if (size_format == 0) {
        // Single stream.
        decode_stream(t, streams, body_size, num_literals, out);
        return out;
    }

    if (body_size < 6) throw ZstdError("huffman jump table truncated");
    std::uint32_t s0 = static_cast<std::uint8_t>(streams[0]) |
                       (static_cast<std::uint8_t>(streams[1]) << 8);
    std::uint32_t s1 = static_cast<std::uint8_t>(streams[2]) |
                       (static_cast<std::uint8_t>(streams[3]) << 8);
    std::uint32_t s2 = static_cast<std::uint8_t>(streams[4]) |
                       (static_cast<std::uint8_t>(streams[5]) << 8);
    std::uint32_t used = 6 + s0 + s1 + s2;
    if (used > body_size) throw ZstdError("huffman jump table overflow");
    std::uint32_t s3 = static_cast<std::uint32_t>(body_size) - used;

    int targets[4];
    stream_sizes(num_literals, targets);
    const std::uint32_t sizes[4] = {s0, s1, s2, s3};
    const std::byte* p = streams + 6;
    for (int i = 0; i < 4; ++i) {
        if (targets[i] > 0) {
            decode_stream(t, p, sizes[i], targets[i], out);
        }
        p += sizes[i];
    }
    return out;
}

}  // namespace fzip::zstd
