# AGENTS.md — build & test commands for fzip

## One-time environment setup (Windows / MSVC 2022)

```pwsh
# From a PowerShell terminal (no need to call vcvars manually; CMake handles it).
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

If Ninja is not autodetecting the compiler, use the Visual Studio generator
instead and skip the rest of the manual environment setup:

```pwsh
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Build

```pwsh
cmake --build build --config Release
```

The `fzip` executable is written to `build/bin/fzip.exe` (Ninja) or
`build/bin/Release/fzip.exe` (VS generator).

## Test (per-stage verification)

Every stage has a round-trip verification that runs `7za t` (CRC test) and
Python `zipfile` checks. The bundled `tests/7za.exe` is used so no system
install is required.

```pwsh
# Run all tests via CTest
ctest --test-dir build --output-on-failure

# Run a single stage's verification directly
python tests/verify.py --stage <N> --binary build/bin/fzip.exe --workdir build/test_tmp
```

## Lint / static checks

There is no separate lint target yet; warnings are treated as errors via
`/WX` (MSVC) or `-Werror` (GCC/Clang) in `CMakeLists.txt`, so a clean build
is the lint gate.

## Submodules

`third_party/zstd` is pinned to tag `v1.5.7`. To restore it after a fresh
clone:

```pwsh
git submodule update --init --recursive
```
