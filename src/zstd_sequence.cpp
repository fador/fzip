// fzip — zstd sequence decoder implementation (RFC 8878 §4.2.2).
#include "zstd_sequence.hpp"

#include <cstring>
#include <stdexcept>

#include "zstd_fse.hpp"
#include "zstd_internal.hpp"

namespace fzip::zstd {

namespace {

// Litlen code → (base, extra_bits). 36 codes total (RFC 8878 §4.2.2).
constexpr int kLitlenBases[] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,18,20,22,24,28,32,40,
    48,64,128,256,512,1024,2048,4096,
    8192,16384,32768,65536,
};
constexpr int kLitlenExtra[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,2,2,3,3,
    4,6,7,8,9,10,11,12,
    13,14,15,16,
};

// Matchlen code → (base, extra_bits). 53 codes total (RFC 8878 §4.2.2).
constexpr int kMatchlenBases[] = {
    3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
    19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
    35,37,39,41,
    43,47,
    51,59,
    67,83,
    99,
    131,
    259,515,1027,2051,4099,8195,16387,32771,65539,
};
constexpr int kMatchlenExtra[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,
    2,2,
    3,3,
    4,4,
    5,
    7,
    8,9,10,11,12,13,14,15,16,
};

}  // namespace

auto litlen_code_to_base(int code) -> int {
    if (code < 0 || code >= 36) return 0;
    return kLitlenBases[code];
}
auto litlen_code_to_extra(int code) -> int {
    if (code < 0 || code >= 36) return 0;
    return kLitlenExtra[code];
}
auto matchlen_code_to_base(int code) -> int {
    if (code < 0 || code >= 53) return 0;
    return kMatchlenBases[code];
}
auto matchlen_code_to_extra(int code) -> int {
    if (code < 0 || code >= 53) return 0;
    return kMatchlenExtra[code];
}

auto offset_code_to_base(int code) -> int {
    if (code < 4) return 0;
    return (1 << (code - 2)) + 1;
}

auto decode_sequences(const std::byte* data, std::size_t size,
                      int num_sequences,
                      const FseSeqSymbol* litlen_table, int ll_acc,
                      const FseSeqSymbol* offset_table, int of_acc,
                      const FseSeqSymbol* matchlen_table, int ml_acc,
                      RepeatOffsets& repeat) -> std::vector<Sequence> {
    if (num_sequences == 0) return {};

    // The FSE bitstream is written forward, read backward from the end.
    FseBitReader reader(data, size);

    // Initial states, read (backward) in order: literals length, offset,
    // match length.
    std::uint32_t ll_state = reader.get_state(ll_acc);
    std::uint32_t of_state = reader.get_state(of_acc);
    std::uint32_t ml_state = reader.get_state(ml_acc);

    std::vector<Sequence> seqs;
    seqs.reserve(static_cast<std::size_t>(num_sequences));
    for (int i = 0; i < num_sequences; ++i) {
        const FseSeqSymbol& ll = litlen_table[ll_state];
        const FseSeqSymbol& ml = matchlen_table[ml_state];
        const FseSeqSymbol& of = offset_table[of_state];

        // Offset. `of.nb_add_bits` is the Offset_Code; codes 0 and 1 are
        // repeat offsets handled exactly per RFC 8878 §3.1.1.5 (matching the
        // reference ZSTD_decodeSequence logic).
        int offset;
        if (of.nb_add_bits > 1) {
            offset = static_cast<int>(of.base_value) +
                     static_cast<int>(reader.read_bits(of.nb_add_bits));
            repeat.offsets[2] = repeat.offsets[1];
            repeat.offsets[1] = repeat.offsets[0];
            repeat.offsets[0] = offset;
        } else {
            const int ll0 = (ll.base_value == 0) ? 1 : 0;
            if (of.nb_add_bits == 0) {
                // Offset_Value 1: RO1 when literals_length > 0, else RO2.
                offset = repeat.offsets[ll0];
                repeat.offsets[1] = repeat.offsets[ll0 ? 0 : 1];
                repeat.offsets[0] = offset;
            } else {
                // Offset_Value 2 or 3.
                const int ov = static_cast<int>(of.base_value) + ll0 +
                               static_cast<int>(reader.read_bits(1));
                int temp;
                if (ov == 3) {
                    temp = repeat.offsets[0] - 1;
                } else {
                    temp = repeat.offsets[ov];
                }
                if (temp == 0) {
                    throw ZstdError("invalid repeat offset (resolves to 0)");
                }
                if (ov != 1) repeat.offsets[2] = repeat.offsets[1];
                repeat.offsets[1] = repeat.offsets[0];
                repeat.offsets[0] = offset = temp;
            }
        }

        int match_len = static_cast<int>(ml.base_value);
        if (ml.nb_add_bits) {
            match_len += static_cast<int>(reader.read_bits(ml.nb_add_bits));
        }
        int lit_len = static_cast<int>(ll.base_value);
        if (ll.nb_add_bits) {
            lit_len += static_cast<int>(reader.read_bits(ll.nb_add_bits));
        }

        Sequence seq;
        seq.literals_length = lit_len;
        seq.match_length = match_len;
        seq.offset = offset;
        seqs.push_back(seq);

        if (i + 1 < num_sequences) {
            ll_state = ll.next_state + reader.read_bits(ll.nb_bits);
            ml_state = ml.next_state + reader.read_bits(ml.nb_bits);
            of_state = of.next_state + reader.read_bits(of.nb_bits);
        }
    }

    return seqs;
}

void execute_sequences(const std::vector<Sequence>& sequences,
                       const std::vector<std::byte>& literals,
                       std::vector<std::byte>& out) {
    int lit_pos = 0;

    for (const auto& seq : sequences) {
        // Copy literal bytes.
        if (seq.literals_length > 0) {
            if (lit_pos + seq.literals_length > static_cast<int>(literals.size())) {
                throw ZstdError("literals overflow in sequence execution");
            }
            out.insert(out.end(), literals.begin() + lit_pos,
                       literals.begin() + lit_pos + seq.literals_length);
            lit_pos += seq.literals_length;
        }

        // Copy match bytes.
        if (seq.match_length > 0) {
            const int offset = seq.offset;
            if (offset <= 0 || static_cast<std::size_t>(offset) > out.size()) {
                throw ZstdError("invalid match offset: " + std::to_string(offset));
            }
            if (seq.match_length > 131074) {
                throw ZstdError("invalid match length: " + std::to_string(seq.match_length));
            }

            std::size_t src = out.size() - static_cast<std::size_t>(offset);
            for (int j = 0; j < seq.match_length; ++j) {
                out.push_back(out[src + static_cast<std::size_t>(j)]);
            }
        }
    }

    // Literals remaining after the last sequence are appended verbatim
    // (RFC 8878 §4.2.5, "Sequence Execution").
    if (lit_pos < static_cast<int>(literals.size())) {
        out.insert(out.end(), literals.begin() + lit_pos, literals.end());
    }
}

}  // namespace fzip::zstd
