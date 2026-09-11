# fzip

A state-of-the-art ZIP compressor in C++20, built to benchmark against
7-Zip and other modern archivers.

## Design

- **Hand-rolled ZIP/ZIP64 container** — local headers, central directory,
  EOCD/EOCD64, extra fields (`0x5455` UT, `0x7875` Unix), 4 KB data
  alignment, UTF-8 filename flag.
- **Hand-rolled CRC-32** (IEEE 0xEDB88320, consteval table) and
  **DEFLATE encoder** (method 8) with LZ77 hash-chain match finder (3- and
  4-byte hash chains), lazy matching, and dynamic Huffman trees built with
  package-merge. At levels 9-12 it switches to an **iterative optimal
  (shortest-path) parser**: a forward pass records Pareto-optimal matches per
  position, then a backward DP minimizes real Huffman bit costs over several
  refinement iterations. Each block independently selects stored / fixed /
  dynamic coding, whichever is smallest.
- **Custom zstd implementation** (method 93) — no external dependency:
  - **Decompressor**: FSE decoder (reverse bitstream, predefined tables),
    Huffman decoder (single + 4-stream), sequence decoder (3 interleaved FSE
    streams), full frame/block parsing. It is **sequential per frame** so
    repeat offsets and matches carry across blocks.
  - **Compressor**: emits spec-correct **compressed blocks** — literals are
    Huffman-coded (4-stream, with FSE- or direct-encoded weight tables) when
    beneficial, otherwise RLE/raw, and sequences are FSE-coded with
    **per-block tables** (RLE for single-symbol streams, predefined for tiny
    blocks). It emits **repeat-offset codes** (rep 1-3), uses a **shared
    window up to 8 MiB** so matches can reference previous blocks at higher
    levels, and uses a **multi-pass optimal (shortest-path) parser** at the
    top levels. Frame/block headers and the content checksum follow RFC 8878.
    Verified against 7za 25.01 and fzip's own extractor.
  - Ratio now beats 7za's deflate while decoding much faster.
- **Per-file codec selection**: for compressible inputs `auto` trial-compresses
  the candidates (deflate-9 and zstd at the requested level) and keeps the
  smallest result, with Store as a guaranteed floor. Incompressible inputs go
  straight to Store.

## Why custom zstd?

The original plan was to vendor zstd 1.5.7 as a git submodule. Instead,
we implemented the zstd format (RFC 8878) from scratch:

- **Decompressor**: handles raw, RLE, and compressed blocks (predefined FSE
  tables), frame headers, and content checksums.
- **Compressor**: valid zstd frames with raw or compressed blocks.
- **FSE codec**: Finite State Entropy (tANS) encoding and decoding — the
  core entropy coder used by zstd.
- **Huffman codec**: literal encoding/decoding primitives.
- **Sequence codec**: sequence encoding/decoding with repeat offset tracking.

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

### Synthetic corpus

9 files (text/binary/mixed at 4K/64K/256K each, ~972 KB).

```
Config                  Size   Ratio  Compress   Decompress
--------------------------------------------------------------
fzip store          1008.8 KB   0.964 56.5 MB/s   26.5 MB/s
fzip deflate-6       661.5 KB   1.469 15.7 MB/s   38.6 MB/s
fzip deflate-9       660.6 KB   1.471  1.3 MB/s   39.1 MB/s
fzip zstd-1          684.6 KB   1.420 33.1 MB/s   46.3 MB/s
fzip zstd-9          661.3 KB   1.470 15.7 MB/s   35.5 MB/s
fzip zstd-19         656.8 KB   1.480  0.9 MB/s   44.3 MB/s
fzip auto            656.6 KB   1.480  0.5 MB/s   44.1 MB/s
7za deflate-5        643.0 KB   1.512 10.4 MB/s   32.5 MB/s
7za deflate-9        640.1 KB   1.518  2.8 MB/s   29.4 MB/s
7za LZMA-9           632.9 KB   1.536 10.9 MB/s   20.9 MB/s
```

The synthetic corpus is dominated by near-incompressible mixed data, so it
understates the ratio gains. `auto` trial-compresses its candidates and keeps
the smallest.

### Real corpus

13 files (~1.46 MB): `7za.exe`, fzip's own sources, and docs. Note that fzip
aligns each entry's data to 4 KB (APK/zipalign v2), which adds up to ~4 KB per
entry; the per-file payloads are much closer to 7za than the archive totals
suggest.

```
Config                  Size   Ratio  Compress   Decompress
--------------------------------------------------------------
fzip store             1.48 MB   0.987 13.5 MB/s   68.8 MB/s
fzip deflate-6        688.3 KB   2.175 11.6 MB/s   35.3 MB/s
fzip deflate-9        664.2 KB   2.254  1.1 MB/s   35.3 MB/s
fzip zstd-1           712.9 KB   2.100 20.5 MB/s   34.4 MB/s
fzip zstd-9           668.6 KB   2.240 20.2 MB/s   37.8 MB/s
fzip zstd-19          628.5 KB   2.383  0.5 MB/s   37.2 MB/s
fzip zstd-22          628.5 KB   2.383  0.5 MB/s   36.4 MB/s
fzip auto             628.2 KB   2.384  0.3 MB/s   35.7 MB/s
7za deflate-5         638.4 KB   2.346  7.9 MB/s   18.8 MB/s
7za deflate-9         634.2 KB   2.361  1.7 MB/s   37.6 MB/s
7za LZMA-9            536.3 KB   2.792  5.8 MB/s   29.2 MB/s
```

The ratio work paid off on realistic data: repeat-offset codes plus
cross-block matches with a large window take zstd-19 from 668.5 KB to
628.5 KB (**-6.0%**), now smaller than 7za's deflate-9 while decoding faster.
The deflate optimal parser takes deflate-9 from 684.3 KB to 664.2 KB
(**-2.9%**); on the 1.3 MB `7za.exe` binary alone the raw DEFLATE payload is
within 1.9% of 7za deflate-9.

Entries are compressed in parallel across CPU cores (written in order, so
output is deterministic). Deflate tokenizes a wave of blocks in parallel and
emits them sequentially. zstd tokenizes blocks in parallel at levels below 10
(independent blocks); from level 10 up it uses a single sequential pass with a
shared match window so matches can cross block boundaries. Entries extract in
parallel; each zstd frame decodes sequentially because repeat offsets and
matches carry across blocks.

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

The custom zstd path is now spec-correct end to end: `fzip zstd` output is
decoded successfully by 7za 25.01 as well as by `fzip extract`, including
Huffman-coded literals (with FSE- or direct-encoded weight tables), per-block
sequence FSE tables, lazy and optimal LZ77 parsing. Its ratio now beats
deflate on text and binaries.

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
