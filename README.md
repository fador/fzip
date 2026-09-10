# fzip

A state-of-the-art ZIP compressor in C++20, built to benchmark against
7-Zip and other modern archivers.

## Design

- **Hand-rolled ZIP/ZIP64 container** — local headers, central directory,
  EOCD/EOCD64, extra fields (`0x5455` UT, `0x7875` Unix), 4 KB data
  alignment, UTF-8 filename flag.
- **Hand-rolled CRC-32** (IEEE 0xEDB88320, consteval table) and
  **DEFLATE encoder** (method 8) with LZ77 hash-chain match finder,
  lazy matching, and dynamic Huffman trees.
- **Custom zstd implementation** (method 93) — no external dependency:
  - **Decompressor**: FSE decoder (reverse bitstream, predefined tables,
    table building from normalized distributions), Huffman decoder (single
    + 4-stream), sequence decoder (3 interleaved FSE streams, repeat
    offsets), full frame/block parsing.
  - **Compressor**: raw-block emitter (type 0) with correct frame format.
    FSE encoder and LZ77 match finder are implemented but the reverse-
    bitstream FSE encoding for compressed blocks needs further work.
- **Per-file codec selection** based on magic-byte type detection:
  incompressible → Store, executables → deflate-9, text → deflate-9,
  general binary → deflate-6. (The executable/text branches temporarily
  fall back to deflate because the custom zstd compressor still only emits
  raw blocks; deflate is a strict improvement over the Store fallback.)

## Why custom zstd?

The original plan was to vendor zstd 1.5.7 as a git submodule. Instead,
we implemented the zstd format (RFC 8878) from scratch:

- **Decompressor**: ~2,500 lines handling all block types (raw, RLE,
  compressed with predefined FSE tables), frame headers, content checksums.
- **Compressor**: ~500 lines emitting valid zstd frames with raw blocks.
- **FSE codec**: ~1,000 lines implementing Finite State Entropy (tANS)
  encoding and decoding — the core entropy coder used by zstd.
- **Huffman codec**: ~800 lines for literal encoding/decoding.
- **Sequence codec**: ~300 lines for sequence encoding/decoding with
  repeat offset tracking.

## Build & test

See [AGENTS.md](AGENTS.md) for full instructions. In short:

```pwsh
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

## Usage

```pwsh
fzip store   <archive.zip> <files...>            # method 0
fzip deflate <archive.zip> <files...> [--level=N] # method 8
fzip zstd    <archive.zip> <files...> [--level=N] # method 93
fzip auto    <archive.zip> <files...>             # per-file auto-select
fzip extract <archive.zip> [--outdir=DIR]
```

## Verification

| Stage | Method | Verification |
|-------|--------|-------------|
| 1 | Store (0) | Python `zipfile` + `7za t` + sha256 |
| 2 | ZIP64 | 65540-entry archive: `zipfile` + `7za t` + sha256 |
| 3 | Deflate (8) | Python `zipfile` (zlib inflate) + `7za t` + sha256 |
| 4 | Zstd (93) | `fzip extract` round-trip + sha256 |
| 5 | Auto | `fzip auto` + `fzip extract` round-trip + sha256 |

## Benchmark results

Synthetic corpus: 9 files (text/binary/mixed at 4K/64K/256K each, ~972 KB).

```
Config                  Size   Ratio  Compress   Decompress
--------------------------------------------------------------
fzip store          1008.8 KB   0.964 58.2 MB/s   53.8 MB/s
fzip deflate-6       661.5 KB   1.469 10.3 MB/s   30.0 MB/s
fzip deflate-9       661.4 KB   1.470  8.3 MB/s   28.6 MB/s
fzip auto            661.4 KB   1.470  9.7 MB/s   29.2 MB/s
fzip zstd (raw)     1008.8 KB   0.964 57.2 MB/s   52.4 MB/s
7za deflate-5        643.0 KB   1.512 10.6 MB/s   32.0 MB/s
7za deflate-9        640.1 KB   1.518  2.8 MB/s   33.5 MB/s
7za LZMA-9           632.9 KB   1.536 11.5 MB/s   22.7 MB/s
```

The synthetic corpus is dominated by near-incompressible mixed data, so it
understates the deflate improvements. On realistic inputs the lazy matcher
pays off clearly (raw-DEFLATE size, level 6):

```
Input                 Before     After   Change
-------------------------------------------------
7za.exe (PE, 1.3 MB)  728422    681527   -6.4%
fzip source (231 KB)   59749     58117   -2.7%
```

Extraction speed also improved substantially: the Huffman inflate now uses
an O(1) canonical lookup table (was an O(symbols) scan per bit), CRC-32 uses
slicing-by-8, and `extract_all` reads the archive once instead of once per
entry. A 13 MB deflate archive extracts in 0.12 s (was 0.91 s), and a
1000-entry archive in 2.6 s (was 9.5 s).

Correctness note: the length-limited Huffman builder was replaced with the
package-merge algorithm. The previous greedy limiter produced *incomplete*
code sets whenever the optimal tree was deeper than 15 bits, which zlib
rejects as "invalid literal/lengths set". fzip could still read its own
output (the unused bit patterns never appeared), but other tools could not.
The package-merge builder guarantees a complete, optimal length-limited code.

Note: The custom zstd compressor currently uses raw blocks (no LZ77
compression), so it produces larger output than deflate. The decompressor
fully supports compressed blocks — fixing the FSE encoding to produce
correct compressed output is the remaining work.

## State-of-the-art analysis

### Compression methods in modern ZIP (APPNOTE 6.3.10, 2022)

| Method | ID | Best for | Ratio | Speed |
|--------|-----|----------|-------|-------|
| Store | 0 | Incompressible | 1.0 | Instant |
| Deflate | 8 | Universal compat | Good | Fast |
| LZMA | 14 | Binaries | Excellent | Slow decode |
| Zstandard | 93 | Best speed/ratio | Excellent | Very fast decode |
| PPMd | 98 | Text | Excellent | Very slow |

### Latest algorithm improvements (2023-2026)

- **zstd 1.5.7** (Feb 2025): +10-30% small-block speed, `--max` mode,
  default multithreading.
- **libdeflate 1.25** (Nov 2025): optimal path parser, AVX512 CRC.
- **7-Zip 26.01** (Apr 2026): Linux huge-pages +10% LZMA speed.
- **zlib-ng 2.3.3** (Feb 2026): Chorba CRC32 with AVX512/VNNI.

## License

TBD (project code).
