# fzip

A state-of-the-art ZIP compressor in C++20, built to benchmark against
7-Zip and other modern archivers.

## Design

- **Hand-rolled ZIP/ZIP64 container** — local headers, central directory,
  EOCD/EOCD64, extra fields (`0x5455` UT, `0x7875` Unix), 4 KB data
  alignment, UTF-8 filename flag.
- **Hand-rolled CRC-32** (IEEE 0xEDB88320) and **DEFLATE encoder** (method 8)
  with an optimal-cost parse — the universal-compatibility tier.
- **Vendored zstd 1.5.7** (method 93) for the ratio tier: levels 1-22,
  `--long=27`, targetCBlockSize, multithreaded encode.
- **Per-file codec selection** based on magic-byte type detection.

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

## Verification

Each stage is verified before commit:

- Store (0) and Deflate (8): Python `zipfile.testzip()` + `zipfile.extractall()`
  + `cmp` against originals. Also `7za t` cross-check.
- Zstd (93) and later: `7za t` (7-Zip 24+ supports zstd-in-zip) + `7za x` +
  `cmp`, since Info-ZIP `unzip` does not understand method 93.

A bundled `tests/7za.exe` is used so no system install is required.

## License

TBD (project code); vendored zstd under its BSD license in
`third_party/zstd/`.
