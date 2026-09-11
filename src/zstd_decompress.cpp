// fzip — zstd decompressor implementation (RFC 8878).
// Stages 1-5: frame parsing, raw/RLE/compressed block decoding.
#include "zstd.hpp"
#include "zstd_internal.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

#include "xxhash.hpp"
#include "zstd_fse.hpp"
#include "zstd_huffman.hpp"
#include "zstd_predefined.hpp"
#include "zstd_sequence.hpp"

namespace fzip::zstd {

namespace {

// --- Frame header parsing (from Stage 1) ---
struct FrameHeader {
    bool single_segment = false;
    bool content_checksum = false;
    bool content_size_present = false;
    std::uint64_t window_size = 0;
    std::uint64_t dictionary_id = 0;
    std::uint64_t frame_content_size = 0;
    bool fcs_unknown = false;
};

auto read_fcs(const std::byte* p, int fcs_field_size) -> std::uint64_t {
    switch (fcs_field_size) {
        case 0: return 0;
        case 1: return static_cast<std::uint64_t>(p[0]);
        case 2: return (static_cast<std::uint64_t>(p[0]) |
                        (static_cast<std::uint64_t>(p[1]) << 8)) + 256;
        case 4: return static_cast<std::uint64_t>(p[0]) |
                       (static_cast<std::uint64_t>(p[1]) << 8) |
                       (static_cast<std::uint64_t>(p[2]) << 16) |
                       (static_cast<std::uint64_t>(p[3]) << 24);
        case 8: {
            std::uint64_t v = 0;
            std::memcpy(&v, p, 8);
            return v;
        }
        default: return 0;
    }
}

auto parse_frame_header(const std::byte* data, std::size_t size,
                        FrameHeader& hdr, std::size_t& bytes_consumed) -> bool {
    if (size < 4) return false;
    std::uint32_t magic;
    std::memcpy(&magic, data, 4);

    if ((magic & 0xFFFFFFF0u) == 0x184D2A50u) {
        if (size < 8) return false;
        std::uint32_t skip_size;
        std::memcpy(&skip_size, data + 4, 4);
        bytes_consumed = 8 + skip_size;
        hdr.fcs_unknown = true;
        return true;
    }

    if (magic != 0xFD2FB528u) {
        throw ZstdError("bad zstd magic: " + std::to_string(magic));
    }

    std::uint8_t desc = static_cast<std::uint8_t>(data[4]);
    int dict_size_code = desc & 0x03;
    static const int dict_sizes[] = {0, 1, 2, 4};
    int dict_id_size = dict_sizes[dict_size_code];

    bool single_segment = (desc >> 5) & 1;
    bool content_checksum = (desc >> 2) & 1;

    int fcs_code = (desc >> 6) & 0x03;
    int fcs_field_size;
    if (fcs_code == 0) {
        fcs_field_size = single_segment ? 1 : 0;
    } else {
        fcs_field_size = 1 << fcs_code;
    }

    hdr.single_segment = single_segment;
    hdr.content_checksum = content_checksum;
    hdr.content_size_present = (fcs_field_size > 0);

    std::size_t pos = 5;

    if (!single_segment) {
        if (pos >= size) return false;
        std::uint8_t wd = static_cast<std::uint8_t>(data[pos++]);
        int window_log = (wd >> 3) + 10;
        std::uint64_t window_base = 1ULL << window_log;
        std::uint64_t window_add = static_cast<std::uint64_t>(wd & 7) << (window_log - 3);
        hdr.window_size = window_base + window_add;
    } else {
        hdr.window_size = 0;
    }

    if (dict_id_size > 0) {
        if (pos + dict_id_size > size) return false;
        hdr.dictionary_id = 0;
        std::memcpy(&hdr.dictionary_id, data + pos, dict_id_size);
        pos += dict_id_size;
    }

    if (fcs_field_size > 0) {
        if (pos + fcs_field_size > size) return false;
        hdr.frame_content_size = read_fcs(data + pos, fcs_field_size);
        pos += fcs_field_size;
    } else {
        hdr.fcs_unknown = true;
    }

    if (single_segment && hdr.frame_content_size > 0) {
        hdr.window_size = hdr.frame_content_size;
    }

    bytes_consumed = pos;
    return true;
}

// --- Block header parsing (from Stage 1) ---
struct BlockHeader {
    bool last_block;
    BlockType type;
    std::uint32_t block_size;
};

auto parse_block_header(const std::byte* data, std::size_t size,
                        BlockHeader& hdr, std::size_t& bytes_consumed) -> bool {
    if (size < 3) return false;
    std::uint32_t raw;
    std::memcpy(&raw, data, 3);
    hdr.last_block = (raw & 1) != 0;
    hdr.type = static_cast<BlockType>((raw >> 1) & 0x03);
    hdr.block_size = (raw >> 3) & 0x1FFFFF;
    bytes_consumed = 3;
    return true;
}

// --- Compressed block decoding (Stages 2-4) ---

// Read a variable-length 1-3 byte integer (used for num_sequences).
auto read_num_sequences(const std::byte* p, const std::byte* end,
                        std::size_t& consumed) -> int {
    if (p >= end) throw ZstdError("num_sequences: truncated");
    std::uint8_t b0 = static_cast<std::uint8_t>(*p++);
    if (b0 < 128) {
        consumed = 1;
        return b0;
    }
    if (b0 < 255) {
        if (p >= end) throw ZstdError("num_sequences: truncated");
        consumed = 2;
        return ((b0 - 128) << 8) + static_cast<std::uint8_t>(*p);
    }
    if (p + 1 >= end) throw ZstdError("num_sequences: truncated");
    consumed = 3;
    return static_cast<std::uint8_t>(p[0]) +
           (static_cast<std::uint8_t>(p[1]) << 8) + 0x7F00;
}

// Decode the literals section (raw, RLE, or Huffman-compressed).
auto decode_literals(const std::byte* data, std::size_t size,
                     std::size_t& consumed) -> std::vector<std::byte> {
    if (size < 1) throw ZstdError("literals block: too short");
    const std::byte* p = data;
    const std::byte* end = data + size;
    std::uint8_t h0 = static_cast<std::uint8_t>(*p);
    int type = h0 & 3;  // 0 = Raw, 1 = RLE, 2 = Compressed, 3 = Treeless
    int size_format = (h0 >> 2) & 3;

    if (type == 0 || type == 1) {
        int regen;
        int hdr_size;
        if (size_format == 0 || size_format == 2) {
            hdr_size = 1;
            regen = h0 >> 3;
        } else if (size_format == 1) {
            if (p + 1 >= end) throw ZstdError("literals header: truncated");
            hdr_size = 2;
            regen = (h0 >> 4) +
                    (static_cast<int>(static_cast<std::uint8_t>(p[1])) << 4);
        } else {
            if (p + 2 >= end) throw ZstdError("literals header: truncated");
            hdr_size = 3;
            regen = (h0 >> 4) +
                    (static_cast<int>(static_cast<std::uint8_t>(p[1])) << 4) +
                    (static_cast<int>(static_cast<std::uint8_t>(p[2])) << 12);
        }
        p += hdr_size;
        if (type == 0) {  // Raw
            if (p + regen > end) throw ZstdError("raw literals: truncated");
            consumed = static_cast<std::size_t>(p - data) +
                       static_cast<std::size_t>(regen);
            return std::vector<std::byte>(p, p + regen);
        }
        // RLE
        if (p >= end) throw ZstdError("rle literals: truncated");
        consumed = static_cast<std::size_t>(p - data) + 1;
        return std::vector<std::byte>(static_cast<std::size_t>(regen), *p);
    }

    if (type == 2) {
        const int k = (size_format == 0 || size_format == 1)
                          ? 10
                          : (size_format == 2 ? 14 : 18);
        const int hdr_size = (4 + 2 * k) / 8;
        if (p + hdr_size > end) throw ZstdError("literals header: truncated");
        std::uint64_t v = 0;
        for (int i = 0; i < hdr_size; ++i) {
            v |= static_cast<std::uint64_t>(
                     static_cast<std::uint8_t>(p[i])) << (8 * i);
        }
        const int regen = static_cast<int>((v >> 4) & ((1u << k) - 1));
        const int csize = static_cast<int>((v >> (4 + k)) & ((1u << k) - 1));
        const std::byte* body = p + hdr_size;
        if (body + csize > end) throw ZstdError("compressed literals: truncated");

        HufTable t;
        std::size_t weight_consumed = 0;
        if (!huf_read_weights(body, static_cast<std::size_t>(csize), t,
                              weight_consumed)) {
            throw ZstdError("unsupported Huffman weight encoding");
        }
        const std::byte* streams = body + weight_consumed;
        const std::size_t streams_size =
            static_cast<std::size_t>(csize) - weight_consumed;
        auto lits = huf_decode_streams(t, streams, streams_size, regen,
                                       size_format);
        consumed = static_cast<std::size_t>(p - data) +
                   static_cast<std::size_t>(hdr_size) +
                   static_cast<std::size_t>(csize);
        return lits;
    }

    throw ZstdError("treeless literals block not supported");
}

// Map a sequence code to its (base, extra-bits) for a given stream kind:
// 0 = literals length, 1 = offset, 2 = match length.
void seq_code_base_extra(int kind, int code, int& base, int& extra) {
    if (kind == 0) {
        base = litlen_code_to_base(code);
        extra = litlen_code_to_extra(code);
    } else if (kind == 1) {
        // Offset: offset = ((1<<code)-3) + read(code) for code >= 2.
        // Codes 0 and 1 are repeat offsets: code 0 has no extra bits, code 1
        // has one (Offset_Value 2 or 3).
        if (code == 0) {
            base = 0;
            extra = 0;
        } else if (code == 1) {
            base = 1;
            extra = 1;
        } else {
            base = static_cast<int>((1u << code) - 3u);
            extra = code;
        }
    } else {
        base = matchlen_code_to_base(code);
        extra = matchlen_code_to_extra(code);
    }
}

// A sequence FSE table resolved for decoding.
struct ResolvedSeqTable {
    std::vector<FseSeqSymbol> storage;
    const FseSeqSymbol* table = nullptr;
    int acc = 0;
};

// Parse one stream's table (predefined / RLE / inline FSE), advancing `p`.
void build_seq_table(int mode, int kind, const std::byte*& p,
                     const std::byte* end, ResolvedSeqTable& out) {
    if (mode == 0) {  // predefined
        if (kind == 0) {
            out.table = kLLDefaultDTable;
            out.acc = 6;
        } else if (kind == 1) {
            out.table = kOFDefaultDTable;
            out.acc = 5;
        } else {
            out.table = kMLDefaultDTable;
            out.acc = 6;
        }
        return;
    }
    if (mode == 1) {  // RLE: one symbol byte
        if (p >= end) throw ZstdError("rle sequence table truncated");
        int code = static_cast<std::uint8_t>(*p++);
        int base, extra;
        seq_code_base_extra(kind, code, base, extra);
        out.storage.resize(1);
        out.storage[0] = {0, static_cast<std::uint8_t>(extra), 0,
                          static_cast<std::uint32_t>(base)};
        out.table = out.storage.data();
        out.acc = 0;
        return;
    }
    // mode 2: inline FSE table description.
    int table_log = 0;
    int max_symbol = 0;
    std::vector<int> norm;
    std::size_t consumed = 0;
    if (!fse_read_ncount(p, static_cast<std::size_t>(end - p), table_log, norm,
                         max_symbol, consumed)) {
        throw ZstdError("invalid sequence FSE table");
    }
    p += consumed;
    FseDecodeTable dt;
    build_fse_dtable(table_log, norm.data(), max_symbol, dt);
    out.storage.resize(dt.symbol.size());
    for (std::size_t u = 0; u < dt.symbol.size(); ++u) {
        int code = dt.symbol[u];
        int base, extra;
        seq_code_base_extra(kind, code, base, extra);
        out.storage[u] = {dt.next_state[u], static_cast<std::uint8_t>(extra),
                          dt.nb_bits[u], static_cast<std::uint32_t>(base)};
    }
    out.table = out.storage.data();
    out.acc = table_log;
}

// Decompress one compressed block, appending to the frame-level `out` so that
// sequences can reference data from previous blocks (and repeat offsets carry
// across compressed blocks).
void decompress_compressed_block(const std::byte* data, std::size_t size,
                                 std::vector<std::byte>& out,
                                 RepeatOffsets& repeat) {
    // Compressed block format:
    //   1. Literals section (variable size)
    //   2. Sequences section

    std::size_t consumed = 0;

    // Decode literals.
    auto literals = decode_literals(data, size, consumed);
    const std::byte* p = data + consumed;
    const std::byte* end = data + size;

    // Read number of sequences.
    std::size_t num_seq_consumed = 0;
    if (p >= end) {
        out.insert(out.end(), literals.begin(), literals.end());
        return;
    }
    int num_sequences = read_num_sequences(p, end, num_seq_consumed);
    p += num_seq_consumed;

    if (num_sequences == 0) {
        out.insert(out.end(), literals.begin(), literals.end());
        return;
    }

    // Symbol compression modes (bits 7-6 LL, 5-4 OF, 3-2 ML, 1-0 reserved).
    if (p >= end) throw ZstdError("sequence symbol types: truncated");
    std::uint8_t modes = static_cast<std::uint8_t>(*p++);
    int ll_mode = (modes >> 6) & 3;
    int of_mode = (modes >> 4) & 3;
    int ml_mode = (modes >> 2) & 3;

    // Resolve each stream's FSE table (predefined / RLE / inline).
    ResolvedSeqTable ll, of, ml;
    build_seq_table(ll_mode, 0, p, end, ll);
    build_seq_table(of_mode, 1, p, end, of);
    build_seq_table(ml_mode, 2, p, end, ml);

    // Decode sequences from the remaining FSE bitstream.
    auto sequences = decode_sequences(p, static_cast<std::size_t>(end - p),
                                      num_sequences, ll.table, ll.acc, of.table,
                                      of.acc, ml.table, ml.acc, repeat);

    // Execute sequences into the frame output.
    execute_sequences(sequences, literals, out);
}

// Decompress one block (handles all block types), appending to `out`.
void decompress_block(const std::byte* data, std::size_t size,
                      const BlockHeader& hdr, std::vector<std::byte>& out,
                      RepeatOffsets& repeat) {
    switch (hdr.type) {
        case BlockType::Raw:
            if (size < hdr.block_size) throw ZstdError("raw block truncated");
            out.insert(out.end(), data, data + hdr.block_size);
            break;
        case BlockType::RLE:
            if (size < 1) throw ZstdError("rle block missing byte");
            out.insert(out.end(), hdr.block_size, data[0]);
            break;
        case BlockType::Compressed:
            decompress_compressed_block(data, size, out, repeat);
            break;
        default:
            throw ZstdError("reserved block type");
    }
}

void verify_checksum(const std::byte* data, std::size_t size,
                     std::span<const std::byte> uncompressed) {
    if (size < 4) throw ZstdError("checksum truncated");
    std::uint32_t stored;
    std::memcpy(&stored, data, 4);
    std::uint64_t computed = xxhash64(uncompressed);
    std::uint32_t computed_lo = static_cast<std::uint32_t>(computed);
    if (stored != computed_lo) {
        throw ZstdError("content checksum mismatch");
    }
}

}  // namespace

auto decompress(std::span<const std::byte> data, std::size_t) -> std::vector<std::byte> {
    if (data.size() < 8) throw ZstdError("zstd data too short");

    const std::byte* p = data.data();
    const std::byte* end = p + data.size();

    FrameHeader hdr;
    std::size_t header_bytes = 0;
    if (!parse_frame_header(p, static_cast<std::size_t>(end - p), hdr, header_bytes)) {
        throw ZstdError("failed to parse frame header");
    }
    p += header_bytes;

    if (hdr.fcs_unknown && !hdr.content_checksum && hdr.frame_content_size == 0) {
        return {};
    }

    // Decode blocks sequentially. Repeat offsets and matches may reference
    // data produced by earlier blocks, so blocks are not independent.
    std::vector<std::byte> output;
    if (!hdr.fcs_unknown && hdr.frame_content_size > 0) {
        output.reserve(static_cast<std::size_t>(hdr.frame_content_size));
    }
    RepeatOffsets repeat;
    bool last = false;
    while (!last && p < end) {
        BlockHeader blk;
        std::size_t blk_bytes = 0;
        if (!parse_block_header(p, static_cast<std::size_t>(end - p), blk, blk_bytes)) {
            throw ZstdError("failed to parse block header");
        }
        p += blk_bytes;
        last = blk.last_block;

        // A block's content is exactly `block_size` bytes (1 byte for RLE),
        // not the rest of the frame — the FSE bitstream finds its sentinel at
        // the end of the block content, so passing trailing bytes (checksum,
        // following blocks) would corrupt decoding.
        const std::size_t content_len =
            (blk.type == BlockType::RLE) ? 1 : blk.block_size;
        const std::size_t avail = static_cast<std::size_t>(end - p);
        decompress_block(p, std::min(content_len, avail), blk, output, repeat);
        p += content_len;
    }

    if (hdr.content_checksum) {
        verify_checksum(p, static_cast<std::size_t>(end - p), output);
    }

    return output;
}

}  // namespace fzip::zstd
