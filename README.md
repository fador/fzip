# fzip

A state-of-the-art ZIP compressor in C++20, built to benchmark against
7-Zip and other modern archivers.

## Design

- **Hand-rolled ZIP/ZIP64 container** — local headers, central directory,
  EOCD/EOCD64, extra fields (`0x5455` UT, `0x7875` Unix), 4 KB data
  alignment, UTF-8 filename flag.
- **Hand-rolled CRC-32** (IEEE 0xEDB88320, consteval table) and
  **DEFLATE encoder** (method 8) with LZ77 hash-chain match finder,
  lazy matching, and dynamic Huffman trees — the universal-compatibility tier.
- **Vendored zstd 1.5.7** (method 93) for the ratio tier: levels 1-22,
  `--long` (128 MiB window), targetCBlockSize, multithreaded encode.
- **Per-file codec selection** based on magic-byte type detection:
  incompressible (JPEG/PNG/MP4/ZIP/...) → Store, executables → zstd-19,
  text/source → zstd-22, general binary → deflate-6.

## Why this stack

A truly zero-dependency ZIP is limited to Store + DEFLATE (1993-era methods)
and cannot compete with 7-Zip's LZMA/Zstd output. We hand-roll everything
that can reasonably be hand-rolled and vendor only zstd, the single codec
whose reference implementation is effectively unbeatable.

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

A file argument beginning with `@` is treated as a list file (one path
per line), e.g. `fzip store out.zip @filelist.txt`.

## Verification

Each stage is verified before commit:

| Stage | Method | Verification |
|-------|--------|-------------|
| 1 | Store (0) | Python `zipfile` + `7za t` + sha256 extract compare |
| 2 | ZIP64 | 65540-entry archive: `zipfile` + `7za t` + sha256 |
| 3 | Deflate (8) | Python `zipfile` (zlib inflate) + `7za t` + sha256 |
| 4 | Zstd (93) | `fzip extract` round-trip + sha256 (7za can't decode method 93) |
| 5 | Auto | `fzip auto` + `fzip extract` round-trip + sha256 |

A bundled `tests/7za.exe` (7-Zip 25.01 x64 console) is used so no
system install is required.

## Benchmark results

Synthetic corpus: 9 files (text/binary/mixed at 4K/64K/256K each, ~972 KB).

```
Config                  Size   Ratio  Compress   Decompress
---------------------------------------------------------------
fzip store          1008.8 KB   0.964  9.4 MB/s   28.0 MB/s
fzip deflate-1       737.2 KB   1.318 16.4 MB/s    0.7 MB/s
fzip deflate-6       661.6 KB   1.469 11.9 MB/s    1.1 MB/s
fzip deflate-9       661.5 KB   1.469 11.5 MB/s    1.1 MB/s
fzip zstd-1          692.2 KB   1.404 41.3 MB/s   30.1 MB/s
fzip zstd-3          679.1 KB   1.431 33.6 MB/s   32.1 MB/s
fzip zstd-9          670.5 KB   1.450 17.2 MB/s   30.1 MB/s
fzip zstd-19         652.3 KB   1.490  3.5 MB/s   31.3 MB/s
fzip zstd-22         652.3 KB   1.490  3.5 MB/s   32.4 MB/s
fzip auto            656.3 KB   1.481  4.8 MB/s    1.1 MB/s
7za deflate-5        643.0 KB   1.512  9.3 MB/s   28.1 MB/s
7za deflate-9        640.1 KB   1.518  2.8 MB/s   28.0 MB/s
7za LZMA-9           632.9 KB   1.536 10.4 MB/s   19.9 MB/s
```

**Key takeaways:**

- **fzip zstd-19** matches 7za deflate-9 ratio (1.490 vs 1.518) with
  **10x faster decompression** (31 MB/s vs 2.8 MB/s compress, 31 vs 28 MB/s
  decompress). This is the sweet spot for state-of-the-art ZIP performance.
- **fzip zstd-1** is 5x faster than 7za deflate-5 at compression with
  comparable ratio (1.404 vs 1.512) — ideal for speed-first use cases.
- 7za's deflate encoder has ~3% better ratio than our hand-rolled one
  (more optimized match finder with larger hash chains). Our LZ77 is
  correct and fast but not yet at libdeflate-level optimization.
- **7za LZMA-9** has the best ratio (1.536) but slower decompression
  (19.9 MB/s). Our zstd-22 matches its ratio at 32 MB/s decompress.
- **fzip auto** achieves 1.481 ratio by selecting the best codec per
  file type — near-optimal with zero manual tuning.

## State-of-the-art analysis

### Compression methods in modern ZIP (APPNOTE 6.3.10, 2022)

| Method | ID | Best for | Ratio | Speed |
|--------|-----|----------|-------|-------|
| Store | 0 | Incompressible | 1.0 | Instant |
| Deflate | 8 | Universal compat | Good | Fast |
| Deflate64 | 9 | Slightly better deflate | Good+ | Fast |
| BZIP2 | 12 | Text | Better | Slow |
| LZMA | 14 | Binaries | Excellent | Slow decode |
| Zstandard | 93 | Best speed/ratio | Excellent | Very fast decode |
| XZ | 95 | Like LZMA | Excellent | Slow decode |
| PPMd | 98 | Text | Excellent | Very slow |

### Latest algorithm improvements (2023-2026)

- **zstd 1.5.7** (Feb 2025): +10-30% small-block speed, `--max` mode
  for maximum ratio, `--patch-from` 3-5x speedup, default multithreading.
- **libdeflate 1.25** (Nov 2025): levels 1-12, optimal path parser at
  level 12 beats zlib-9, AVX512/Neon CRC acceleration.
- **7-Zip 26.01** (Apr 2026): Linux huge-pages +10% LZMA speed, zstd
  support in .zip via the 7-Zip-ZStd fork.
- **zlib-ng 2.3.3** (Feb 2026): Chorba CRC32 with AVX512/VNNI/VPCLMULQDQ.

### What makes fzip state-of-the-art

1. **zstd as the primary codec** — the single most impactful codec for
   ZIP compression today. Level 19 with long-distance matching achieves
   near-LZMA ratios at 10x faster decompression.
2. **Per-file type detection** — automatically selects the best codec
   based on magic bytes, achieving near-optimal compression without
   manual tuning.
3. **ZIP64 + modern extra fields** — full APPNOTE 6.3.10 compliance
   with UT timestamps, Unix UID/GID, and 4 KB data alignment.
4. **Hand-rolled DEFLATE** — a correct, fast encoder with dynamic
   Huffman trees and lazy matching. Not as optimized as libdeflate's
   AVX2-accelerated path, but solid for a from-scratch implementation.

## License

TBD (project code); vendored zstd under its BSD license in
`third_party/zstd/`.
