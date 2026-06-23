// fzip — zstd decompressor implementation (RFC 8878).
// Stage 1: frame + block header parsing, raw/RLE blocks, skippable frames.
// Stages 2-4: compressed block decoding (FSE, Huffman, sequences).
#include "zstd.hpp"
#include "zstd_internal.hpp"

#include <cstring>

#include "xxhash.hpp"

namespace fzip::zstd {

namespace {

// --- Frame header parsing ---
// Frame header layout: Magic(4) + Descriptor(1) + [Window_Desc(1)] +
// [Dictionary_ID(0-4)] + [Frame_Content_Size(0-8)].
struct FrameHeader {
    bool single_segment = false;
    bool content_checksum = false;
    bool content_size_present = false;
    std::uint64_t window_size = 0;
    std::uint64_t dictionary_id = 0;
    std::uint64_t frame_content_size = 0;
    bool fcs_unknown = false;  // FCS field absent
};

auto read_fcs(const std::byte* p, int fcs_field_size) -> std::uint64_t {
    switch (fcs_field_size) {
        case 0: return 0;  // not present
        case 1: return static_cast<std::uint64_t>(p[0]) + 256;
        case 2: return static_cast<std::uint64_t>(p[0]) |
                       (static_cast<std::uint64_t>(p[1]) << 8);
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
                        FrameHeader& hdr, std::size_t& bytes_consumed)
    -> bool {
    if (size < 4) return false;

    // Magic number.
    std::uint32_t magic;
    std::memcpy(&magic, data, 4);

    // Check for skippable frame.
    if ((magic & 0xFFFFFFF0u) == kSkippableMagicBase) {
        // Skippable frame: magic(4) + size(4) + data[size].
        if (size < 8) return false;
        std::uint32_t skip_size;
        std::memcpy(&skip_size, data + 4, 4);
        bytes_consumed = 8 + skip_size;
        hdr.fcs_unknown = true;
        return true;
    }

    if (magic != kMagic) {
        throw ZstdError("bad zstd magic: " + std::to_string(magic));
    }

    // Frame header descriptor (byte 4).
    std::uint8_t desc = static_cast<std::uint8_t>(data[4]);

    // Bit layout (RFC 8878 §3.1.1.1):
    //   Bit 0-1: Dictionary_ID_Size code {0,1,2,3} → {0,1,2,4} bytes
    //   Bit 2:   Content_Checksum_Flag
    //   Bit 3:   Reserved (0)
    //   Bit 4:   Unused
    //   Bit 5:   Single_Segment_flag
    //   Bit 6-7: FCS_Field_Size code
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
        fcs_field_size = 1 << (fcs_code + 1);  // 1→2, 2→4, 3→8
    }

    hdr.single_segment = single_segment;
    hdr.content_checksum = content_checksum;
    hdr.content_size_present = (fcs_field_size > 0);

    std::size_t pos = 5;  // after magic + descriptor

    // Window descriptor (1 byte, present if !single_segment).
    if (!single_segment) {
        if (pos >= size) return false;
        std::uint8_t wd = static_cast<std::uint8_t>(data[pos++]);
        // Window size = ((2^((wd >> 3) + 10)) + ((wd & 7) << ((wd >> 3) + 8)))
        // Actually per RFC 8878 §3.1.1.1.2:
        //   window_log = (wd >> 3) + 10
        //   window_base = 1 << window_log
        //   window_add = (wd & 7) << (window_log - 3)
        //   window_size = window_base + window_add
        int window_log = (wd >> 3) + 10;
        std::uint64_t window_base = 1ULL << window_log;
        std::uint64_t window_add = static_cast<std::uint64_t>(wd & 7) << (window_log - 3);
        hdr.window_size = window_base + window_add;
    } else {
        // Single segment: window size = frame content size (if known).
        hdr.window_size = 0;  // will be set after FCS is read
    }

    // Dictionary ID (0-4 bytes).
    if (dict_id_size > 0) {
        if (pos + dict_id_size > size) return false;
        hdr.dictionary_id = 0;
        std::memcpy(&hdr.dictionary_id, data + pos, dict_id_size);
        pos += dict_id_size;
    }

    // Frame content size (0-8 bytes).
    if (fcs_field_size > 0) {
        if (pos + fcs_field_size > size) return false;
        hdr.frame_content_size = read_fcs(data + pos, fcs_field_size);
        pos += fcs_field_size;
    } else {
        hdr.fcs_unknown = true;
    }

    // For single segment, window size = content size.
    if (single_segment && hdr.frame_content_size > 0) {
        hdr.window_size = hdr.frame_content_size;
    }

    // Frame header must be a multiple of 4 bytes total (magic + header).
    // Actually: the frame header size (including magic) must be a multiple
    // of 4 bytes. We need to check and skip padding if necessary.
    // Wait — no, the frame header descriptor includes a padding bit field.
    // Actually per RFC: the frame header is NOT padded to 4 bytes. The
    // total frame header size is 4 (magic) + 1 (descriptor) + optional
    // window desc + optional dict ID + optional FCS. No padding.

    bytes_consumed = pos;
    return true;
}

// --- Block parsing ---
struct BlockHeader {
    bool last_block;
    BlockType type;
    std::uint32_t block_size;
};

auto parse_block_header(const std::byte* data, std::size_t size,
                        BlockHeader& hdr, std::size_t& bytes_consumed)
    -> bool {
    if (size < 3) return false;
    std::uint32_t raw;
    std::memcpy(&raw, data, 3);
    // 3 bytes in little-endian: bit 0 = last-block, bits 1-2 = type,
    // bits 3-23 = block size.
    hdr.last_block = (raw & 1) != 0;
    hdr.type = static_cast<BlockType>((raw >> 1) & 0x03);
    hdr.block_size = (raw >> 3) & 0x1FFFFF;  // 21 bits
    bytes_consumed = 3;
    return true;
}

// Decompress one block. Returns the uncompressed bytes.
// For Stage 1: handles Raw and RLE blocks only.
auto decompress_block(const std::byte* data, std::size_t size,
                      const BlockHeader& hdr) -> std::vector<std::byte> {
    switch (hdr.type) {
        case BlockType::Raw: {
            if (size < hdr.block_size) {
                throw ZstdError("raw block truncated");
            }
            return std::vector<std::byte>(data, data + hdr.block_size);
        }
        case BlockType::RLE: {
            if (size < 1) throw ZstdError("rle block missing byte");
            std::vector<std::byte> out(hdr.block_size, data[0]);
            return out;
        }
        case BlockType::Compressed: {
            // Stage 2-4 will implement this.
            throw ZstdError("compressed blocks not yet implemented (Stage 1)");
        }
        default:
            throw ZstdError("reserved block type");
    }
}

// Read and verify the content checksum (xxHash-64, low 32 bits).
void verify_checksum(const std::byte* data, std::size_t size,
                     std::span<const std::byte> uncompressed) {
    if (size < 4) throw ZstdError("checksum truncated");
    std::uint32_t stored;
    std::memcpy(&stored, data, 4);
    std::uint64_t computed = xxhash64(uncompressed);
    std::uint32_t computed_lo = static_cast<std::uint32_t>(computed);
    if (stored != computed_lo) {
        throw ZstdError("content checksum mismatch: stored=" +
                        std::to_string(stored) +
                        " computed=" + std::to_string(computed_lo));
    }
}

}  // namespace

auto decompress(std::span<const std::byte> data, std::size_t /*expected_size*/)
    -> std::vector<std::byte> {
    if (data.size() < 8) {
        throw ZstdError("zstd data too short");
    }

    const std::byte* p = data.data();
    const std::byte* end = p + data.size();

    // Parse frame header.
    FrameHeader hdr;
    std::size_t header_bytes = 0;
    if (!parse_frame_header(p, static_cast<std::size_t>(end - p), hdr,
                            header_bytes)) {
        throw ZstdError("failed to parse frame header");
    }
    p += header_bytes;

    // If this was a skippable frame, just return empty.
    if (hdr.fcs_unknown && !hdr.content_checksum &&
        hdr.frame_content_size == 0) {
        return {};
    }

    // Decompress blocks.
    std::vector<std::byte> output;
    // Pre-allocate if content size is known.
    if (!hdr.fcs_unknown && hdr.frame_content_size > 0) {
        output.reserve(static_cast<std::size_t>(hdr.frame_content_size));
    }

    while (p < end) {
        BlockHeader blk;
        std::size_t blk_bytes = 0;
        if (!parse_block_header(p, static_cast<std::size_t>(end - p),
                                blk, blk_bytes)) {
            throw ZstdError("failed to parse block header");
        }
        p += blk_bytes;

        auto block_data = decompress_block(p, static_cast<std::size_t>(end - p),
                                           blk);
        output.insert(output.end(), block_data.begin(), block_data.end());
        p += (blk.type == BlockType::RLE) ? 1 : blk.block_size;

        if (blk.last_block) break;
    }

    // Verify content checksum if present.
    if (hdr.content_checksum) {
        verify_checksum(p, static_cast<std::size_t>(end - p), output);
    }

    return output;
}

}  // namespace fzip::zstd
