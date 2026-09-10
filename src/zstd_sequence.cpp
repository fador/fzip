// fzip — zstd sequence decoder implementation (RFC 8878 §4.2.2).
#include "zstd_sequence.hpp"

#include <cstring>
#include <stdexcept>

#include "zstd_fse.hpp"
#include "zstd_internal.hpp"

namespace fzip::zstd {

namespace {

// Litlen code → (base, extra_bits). 36 codes total.
constexpr int kLitlenBases[] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,18,20,22,24,26,28,30,
    32,36,40,44,
    48,56,64,72,
    80,96,112,128,
};
constexpr int kLitlenExtra[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,
    2,2,2,2,
    3,3,3,3,
    4,4,4,4,
};

// Matchlen code → (base, extra_bits). 53 codes total.
// Codes 0-31: base=code+3, extra=0
// Codes 32-35: base=35+(c-32)*2, extra=1
// Codes 36-39: base=43+(c-36)*4, extra=2
// Codes 40-43: base=59+(c-40)*8, extra=3
// Codes 44-47: base=91+(c-44)*16, extra=4
// Codes 48-52: base=155+(c-48)*32, extra=5
constexpr int kMatchlenBases[] = {
    3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
    19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
    35,37,39,41,
    43,47,51,55,
    59,67,75,83,
    91,107,123,139,
    155,187,219,251,283,
};
constexpr int kMatchlenExtra[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,
    2,2,2,2,
    3,3,3,3,
    4,4,4,4,
    5,5,5,5,5,
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
                      const FseSeqSymbol* litlen_table,
                      const FseSeqSymbol* offset_table,
                      const FseSeqSymbol* matchlen_table)
    -> std::vector<Sequence> {
    if (num_sequences == 0) return {};

    // The FSE bitstream is written forward, read backward from the end.
    FseBitReader reader(data, size);

    // Initial states, read (backward) in order: literals length, offset,
    // match length.
    std::uint32_t ll_state = reader.get_state(6);
    std::uint32_t of_state = reader.get_state(5);
    std::uint32_t ml_state = reader.get_state(6);

    std::vector<Sequence> seqs;
    seqs.reserve(static_cast<std::size_t>(num_sequences));
    for (int i = 0; i < num_sequences; ++i) {
        const FseSeqSymbol& ll = litlen_table[ll_state];
        const FseSeqSymbol& ml = matchlen_table[ml_state];
        const FseSeqSymbol& of = offset_table[of_state];

        // Decoder consumes: offset extra, match-len extra, lit-len extra,
        // then updates states LL, ML, OF.
        if (of.nb_add_bits <= 1) {
            throw ZstdError("repeat offset codes are not supported");
        }
        int offset = static_cast<int>(of.base_value) +
                     static_cast<int>(reader.read_bits(of.nb_add_bits));
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

auto execute_sequences(const std::vector<Sequence>& sequences,
                       const std::vector<std::byte>& literals,
                       RepeatOffsets& repeat) -> std::vector<std::byte> {
    std::vector<std::byte> out;
    out.reserve(literals.size() * 2);

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
            int offset = seq.offset;

            // Resolve repeat offsets (encoded as negative values).
            if (offset < 0) {
                int raw_code = -(offset) - 1;  // 0, 1, 2, or 3
                if (raw_code == 0) {
                    // Code 0: use repeat[0], no update.
                    offset = repeat.offsets[0];
                } else if (raw_code == 1) {
                    // Code 1: use repeat[1], swap repeat[0] and repeat[1].
                    offset = repeat.offsets[1];
                    std::swap(repeat.offsets[0], repeat.offsets[1]);
                } else if (raw_code == 2) {
                    // Code 2: use repeat[2], rotate repeat[].
                    offset = repeat.offsets[2];
                    int tmp = repeat.offsets[2];
                    repeat.offsets[2] = repeat.offsets[1];
                    repeat.offsets[1] = repeat.offsets[0];
                    repeat.offsets[0] = tmp;
                } else {
                    // Code 3: if litlen == 0 → offset = repeat[0] - 1
                    //          else → offset = 1
                    if (seq.literals_length == 0) {
                        offset = repeat.offsets[0] - 1;
                    } else {
                        offset = 1;
                    }
                    // No repeat update for code 3.
                }
            } else {
                // Normal offset (code >= 4): update repeat offsets.
                repeat.offsets[2] = repeat.offsets[1];
                repeat.offsets[1] = repeat.offsets[0];
                repeat.offsets[0] = offset;
            }

            // Copy match bytes (may overlap with output).
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

    return out;
}

}  // namespace fzip::zstd
