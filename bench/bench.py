#!/usr/bin/env python3
"""Benchmark fzip vs 7za on representative test data.

Creates synthetic test files (text, binary, mixed) and measures:
  - Compressed size and ratio for each codec/level
  - Compression and decompression speed

Usage:
    python bench/bench.py --binary build/bin/fzip.exe [--workdir bench/tmp]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path


def run_timed(cmd: list[str], timeout: float = 60) -> tuple[float, int, str]:
    """Run a command, return (elapsed_seconds, returncode, stderr)."""
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.perf_counter() - t0
        return elapsed, r.returncode, r.stderr
    except subprocess.TimeoutExpired:
        return timeout, -1, "TIMEOUT"
    except Exception as e:
        return 0.0, -1, str(e)


def file_size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_text(size: int) -> bytes:
    """Generate compressible English-like text."""
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "hello", "world", "zip", "compress", "data", "file", "test",
        "algorithm", "state", "art", "performance", "ratio", "speed",
        "deflate", "zstd", "lzma", "bzip2", "huffman", "lz77", "match",
        "window", "dictionary", "encoder", "decoder", "stream", "block",
        "binary", "format", "standard", "compatible", "verify", "round",
        "trip", "benchmark", "corpus", "analysis", "improvement", "modern",
    ]
    rng = random.Random(42)
    parts = []
    while len(" ".join(parts).encode()) < size:
        parts.append(rng.choice(words))
    text = " ".join(parts)[:size]
    return text.encode("utf-8")


def generate_binary(size: int) -> bytes:
    """Generate semi-structured binary (executable-like)."""
    rng = random.Random(123)
    # Mix of structured patterns and random bytes.
    data = bytearray()
    while len(data) < size:
        # Common pattern: x86-like call/jump sequences.
        data.extend(b"\xE8\x00\x00\x00\x00")  # call $+5
        data.extend(b"\x48\x89\xE5")            # mov rbp, rsp
        data.extend(rng.randbytes(rng.randint(10, 100)))
        data.extend(b"\xC3")                     # ret
    return bytes(data[:size])


def generate_mixed(size: int) -> bytes:
    """Generate mixed content (JSON-like structured text + binary blobs)."""
    rng = random.Random(777)
    parts = []
    while len(b"".join(parts)) < size:
        if rng.random() < 0.7:
            # JSON-like text
            key = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(3, 15)))
            val = rng.randint(0, 100000)
            parts.append(f'{{"{key}": {val}}}\n'.encode())
        else:
            # Binary blob
            parts.append(rng.randbytes(rng.randint(20, 200)))
    return b"".join(parts)[:size]


def make_corpus(workdir: Path) -> list[tuple[str, int]]:
    """Create test corpus files. Returns list of (filename, size)."""
    corpus_dir = workdir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    return _make_synthetic_corpus(corpus_dir)


def _make_synthetic_corpus(corpus_dir: Path) -> list[tuple[str, int]]:

    files = [
        ("text_4k.txt", 4 * 1024, generate_text),
        ("text_64k.txt", 64 * 1024, generate_text),
        ("text_256k.txt", 256 * 1024, generate_text),
        ("binary_4k.bin", 4 * 1024, generate_binary),
        ("binary_64k.bin", 64 * 1024, generate_binary),
        ("binary_256k.bin", 256 * 1024, generate_binary),
        ("mixed_4k.mix", 4 * 1024, generate_mixed),
        ("mixed_64k.mix", 64 * 1024, generate_mixed),
        ("mixed_256k.mix", 256 * 1024, generate_mixed),
    ]

    result = []
    for name, size, gen in files:
        p = corpus_dir / name
        if not p.exists():
            p.write_bytes(gen(size))
        result.append((name, size))
    return result


def use_real_corpus(workdir: Path, src_dir: Path,
                    max_files: int) -> list[tuple[str, int]]:
    """Copy up to `max_files` files from `src_dir` into the corpus dir.

    Files are flattened with an index prefix to avoid basename collisions.
    Returns the list of (filename, size)."""
    corpus_dir = workdir / "corpus"
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(p for p in src_dir.rglob("*") if p.is_file())
    # Prefer larger, more representative files first.
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    result: list[tuple[str, int]] = []
    for i, p in enumerate(candidates[:max_files]):
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if not data:
            continue
        name = f"{i:04d}_{p.name}"
        (corpus_dir / name).write_bytes(data)
        result.append((name, len(data)))
    return result


def bench_fzip(binary: Path, corpus_dir: Path, archive: Path,
               mode: str, extra: list[str]) -> tuple[int, float, float]:
    """Compress with fzip, return (compressed_size, compress_time, decompress_time)."""
    files = sorted(corpus_dir.iterdir())
    file_args = [str(f) for f in files if f.is_file()]
    if not file_args:
        return 0, 0.0, 0.0

    # Compress.
    cmd = [str(binary), mode, str(archive)] + file_args + extra
    comp_time, rc, stderr = run_timed(cmd)
    if rc != 0:
        print(f"  fzip {mode} FAILED: {stderr[-200:]}")
        return 0, comp_time, 0.0

    comp_size = file_size(archive)

    # Decompress (extract).
    extract_dir = archive.parent / "extract_bench"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    cmd = [str(binary), "extract", str(archive), f"--outdir={extract_dir}"]
    decomp_time, rc, stderr = run_timed(cmd)
    if rc != 0:
        print(f"  fzip extract FAILED: {stderr[-200:]}")
        return comp_size, comp_time, decomp_time

    # Verify.
    for f in files:
        if not f.is_file():
            continue
        extracted = extract_dir / f.name
        if not extracted.exists():
            print(f"  MISSING: {f.name}")
            continue
        if sha256(f) != sha256(extracted):
            print(f"  MISMATCH: {f.name}")

    shutil.rmtree(extract_dir, ignore_errors=True)
    return comp_size, comp_time, decomp_time


def bench_7za(seven_zip: Path, corpus_dir: Path, archive: Path,
              switches: list[str]) -> tuple[int, float, float]:
    """Compress with 7za, return (compressed_size, compress_time, decompress_time)."""
    files = sorted(corpus_dir.iterdir())
    file_args = [str(f) for f in files if f.is_file()]
    if not file_args:
        return 0, 0.0, 0.0

    # Compress.
    cmd = [str(seven_zip), "a"] + switches + [str(archive)] + file_args
    comp_time, rc, stderr = run_timed(cmd)
    if rc != 0:
        print(f"  7za FAILED: {stderr[-200:]}")
        return 0, comp_time, 0.0

    comp_size = file_size(archive)

    # Decompress (test).
    extract_dir = archive.parent / "extract_7za"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    cmd = [str(seven_zip), "x", f"-o{extract_dir}", str(archive)]
    decomp_time, rc, stderr = run_timed(cmd)
    if rc != 0:
        print(f"  7za x FAILED: {stderr[-200:]}")

    shutil.rmtree(extract_dir, ignore_errors=True)
    return comp_size, comp_time, decomp_time


def format_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def format_speed(size_bytes: int, elapsed: float) -> str:
    if elapsed <= 0:
        return "N/A"
    mbps = size_bytes / elapsed / (1024 * 1024)
    return f"{mbps:.1f} MB/s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--seven-zip", type=Path, default=None,
                    help="Path to 7za.exe (default: tests/7za.exe)")
    ap.add_argument("--workdir", type=Path, default=Path("bench/tmp"))
    ap.add_argument("--corpus-dir", type=Path, default=None,
                    help="Use real files from this directory instead of the "
                         "synthetic corpus (recursively, flattened, up to "
                         "--max-files).")
    ap.add_argument("--max-files", type=int, default=64)
    args = ap.parse_args()

    if args.seven_zip is None:
        args.seven_zip = Path(__file__).resolve().parent.parent / "tests" / "7za.exe"

    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.corpus_dir is not None:
        corpus_files = use_real_corpus(args.workdir, args.corpus_dir,
                                       args.max_files)
    else:
        corpus_files = make_corpus(args.workdir)
    corpus_dir = args.workdir / "corpus"
    total_input = sum(sz for _, sz in corpus_files)

    print(f"Corpus: {len(corpus_files)} files, {format_size(total_input)} total")
    print()

    # Define benchmark configurations.
    configs: list[tuple[str, str, list[str]]] = []  # (name, tool, args)
    configs.append(("fzip store", "fzip", ["store"]))
    configs.append(("fzip deflate-1", "fzip", ["deflate", "--level=1"]))
    configs.append(("fzip deflate-6", "fzip", ["deflate", "--level=6"]))
    configs.append(("fzip deflate-9", "fzip", ["deflate", "--level=9"]))
    configs.append(("fzip zstd-1", "fzip", ["zstd", "--level=1"]))
    configs.append(("fzip zstd-3", "fzip", ["zstd", "--level=3"]))
    configs.append(("fzip zstd-9", "fzip", ["zstd", "--level=9"]))
    configs.append(("fzip zstd-19", "fzip", ["zstd", "--level=19"]))
    configs.append(("fzip zstd-22", "fzip", ["zstd", "--level=22"]))
    configs.append(("fzip auto", "fzip", ["auto"]))
    configs.append(("7za deflate-5", "7za", ["-tzip", "-mx=5"]))
    configs.append(("7za deflate-9", "7za", ["-tzip", "-mx=9"]))
    configs.append(("7za LZMA-9", "7za", ["-tzip", "-mx=9", "-m0=LZMA"]))

    results = []
    for name, tool, extra in configs:
        archive = args.workdir / f"{name.replace(' ', '_').replace('-', '_')}.zip"
        if tool == "fzip":
            mode = extra[0]
            mode_extra = extra[1:] if len(extra) > 1 else []
            sz, ct, dt = bench_fzip(args.binary, corpus_dir, archive, mode, mode_extra)
        else:
            sz, ct, dt = bench_7za(args.seven_zip, corpus_dir, archive, extra)
        ratio = total_input / sz if sz > 0 else 0
        results.append((name, sz, ratio, ct, dt))
        if archive.exists():
            archive.unlink()

    # Print results table.
    print(f"{'Config':<22} {'Size':>10} {'Ratio':>7} {'Compress':>12} {'Decompress':>12}")
    print("-" * 65)
    for name, sz, ratio, ct, dt in results:
        size_str = format_size(sz) if sz > 0 else "FAILED"
        ratio_str = f"{ratio:.3f}" if ratio > 0 else "N/A"
        cspeed = format_speed(total_input, ct) if ct > 0 else "N/A"
        dspeed = format_speed(total_input, dt) if dt > 0 else "N/A"
        print(f"{name:<22} {size_str:>10} {ratio_str:>7} {cspeed:>12} {dspeed:>12}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
