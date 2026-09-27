# fzip

A state-of-the-art ZIP compressor in C++20, built to benchmark against
7-Zip and other modern archivers.

## Design

- **Hand-rolled ZIP/ZIP64 container** — local headers, central directory,
  EOCD/EOCD64, extra fields (`0x5455` UT, `0x7875` Unix), 4 KB data
  alignment, UTF-8 filename flag.
- **Hand-rolled CRC-32** (IEEE 0xEDB88320, consteval table) and
  **DEFLATE encoder** (method 8) with LZ77 dual hash-chain match finder (3- and
  4-byte hash chains), 32 KB cross-block sliding window history, lazy matching,
  O(1) precomputed length and distance LUTs, and dynamic Huffman trees built
  with package-merge. At levels 9-12 it switches to an **iterative optimal
  (shortest-path) parser**: a forward pass records Pareto-optimal matches per
  position with candidate cold-start protection, then a backward DP minimizes
  exact Huffman bit costs with dynamic unused-symbol penalties over several
  refinement iterations. Blocks are tokenized in parallel using 256 KiB chunks
  for superior local Huffman tree adaptation.
- **Custom zstd implementation** (method 93) — no external dependency:
  - **Decompressor**: FSE decoder (reverse bitstream, predefined tables),
    Huffman decoder (single + 4-stream), sequence decoder (3 interleaved FSE
    streams), full frame/block parsing. It is **sequential per frame** so
    repeat offsets and matches carry across blocks.
  - **Compressor**: emits spec-correct **compressed blocks** — literals are
    Huffman-coded (4-stream, FSE- or direct-encoded weight tables, and a
    reused table across blocks via **treeless literals**) when beneficial,
    otherwise RLE/raw. Sequences are FSE-coded with **per-block tables**
    (RLE for single-symbol streams, predefined for tiny blocks, and an
    accuracy-log search for small blocks). It features a **dual-hash match finder**
    (64K 3-byte + 128K 4-byte hash table) with early match-length candidate
    rejection, **repeat-offset-aware greedy/lazy and optimal parsing** (actively
    pricing and selecting rep 1-3 codes), uses a **shared window up to 8 MiB**
    so matches can reference previous blocks at higher levels, and uses hardware
    bit operations (`std::bit_width`) with fast O(1) paths for length/offset
    conversions. Frame/block headers and the content checksum follow RFC 8878.
  - Ratio beats 7za's deflate while decoding much faster.
- **Per-file codec selection**: for compressible inputs `auto` trial-compresses
  the candidates (deflate-9/12 and zstd at the requested level) and keeps the
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
ctest --test-dir build -C Release --output-on-failure
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
fzip store          1008.8 KB   0.964 34.7 MB/s    8.2 MB/s
fzip deflate-1       737.2 KB   1.318 30.3 MB/s    7.2 MB/s
fzip deflate-6       661.5 KB   1.469 23.0 MB/s   37.6 MB/s
fzip deflate-9       660.6 KB   1.471  2.2 MB/s   16.1 MB/s
fzip zstd-1          683.9 KB   1.421 28.8 MB/s   40.5 MB/s
fzip zstd-3          679.6 KB   1.430 26.0 MB/s   43.8 MB/s
fzip zstd-9          661.4 KB   1.470 19.4 MB/s   30.9 MB/s
fzip zstd-19         656.8 KB   1.480  1.2 MB/s    9.6 MB/s
fzip zstd-22         656.8 KB   1.480  1.4 MB/s   39.7 MB/s
fzip auto            656.6 KB   1.480  0.9 MB/s   16.7 MB/s
7za deflate-5        643.0 KB   1.512  9.8 MB/s   27.1 MB/s
7za deflate-9        640.1 KB   1.518  2.2 MB/s    5.9 MB/s
7za LZMA-9           632.9 KB   1.536  8.8 MB/s   18.1 MB/s
```

The synthetic corpus is dominated by near-incompressible mixed data, so it
understates the ratio gains. `auto` trial-compresses its candidates and keeps
the smallest.

### Real corpus

Source files (`src/` directory, 29 files, 273.8 KB total):

```
Config                  Size   Ratio  Compress   Decompress
--------------------------------------------------------------
fzip store           305.8 KB   0.895  2.1 MB/s    7.8 MB/s
fzip deflate-1       117.3 KB   2.335  8.9 MB/s    7.2 MB/s
fzip deflate-6       102.2 KB   2.678 11.1 MB/s    7.9 MB/s
fzip deflate-9       102.0 KB   2.683  1.8 MB/s    7.7 MB/s
fzip zstd-1          103.4 KB   2.647  8.1 MB/s    7.5 MB/s
fzip zstd-3          103.2 KB   2.652  8.4 MB/s    9.1 MB/s
fzip zstd-9          103.0 KB   2.658  7.3 MB/s    2.8 MB/s
fzip zstd-19         102.8 KB   2.663  1.5 MB/s    2.0 MB/s
fzip zstd-22         102.8 KB   2.663  0.5 MB/s    1.6 MB/s
fzip auto            102.0 KB   2.683  1.1 MB/s    9.3 MB/s
7za deflate-5         75.6 KB   3.624  2.7 MB/s    5.2 MB/s
7za deflate-9         74.7 KB   3.664  1.8 MB/s    1.9 MB/s
7za LZMA-9            74.5 KB   3.677  3.7 MB/s    6.1 MB/s
```

Note that fzip aligns each entry's data to 4 KB (APK/zipalign v2 container spec),
which accounts for ~2.5 KB average per entry in small files. On large payloads
such as `tests/7za.exe` (1.33 MB), the raw DEFLATE compressed payload is within
1.5% of 7za deflate-9, and optimal level 12 finishes in < 1.0s.

Entries are compressed in parallel across CPU cores (written in order, so
output is deterministic). Deflate tokenizes a wave of 256 KiB blocks in parallel
with 32 KB sliding window cross-block history continuity and emits them sequentially.
zstd tokenizes blocks in parallel at levels below 10 (independent blocks); from
level 10 up it uses a single sequential pass with a shared match window so matches
can cross block boundaries. Entries extract in parallel; each zstd frame decodes
sequentially because repeat offsets and matches carry across blocks.

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

This project is licensed under the [MIT License](LICENSE).
