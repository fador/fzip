#!/usr/bin/env python3
"""fzip per-stage verification harness.

Runs the appropriate round-trip check for a given stage using:
  * Python's built-in `zipfile` module (Store + Deflate methods),
  * the bundled `tests/7za.exe` for methods Info-ZIP `unzip` can't handle
    (e.g. Zstd method 93), and
  * byte-for-byte comparison of extracted files against the originals.

Usage:
    python tests/verify.py --stage <N> --binary <path> --workdir <dir>

Stages:
    0 — smoke: binary exists and `--version` prints a version string.
    1 — Store: fzip store writes a .zip; zipfile + 7za verify it.
    2 — ZIP64: build an archive with >65535 entries; 7za + zipfile verify.
    3 — Deflate: fzip deflate; zipfile.testzip() + 7za t + cmp.
    4 — Zstd: fzip zstd; 7za t (zstd-in-zip) + 7za x + cmp.
    5 — Auto: per-file codec selection; 7za t + 7za x + cmp.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_7za(tests_dir: Path) -> Path:
    # tests/7za.exe is bundled. If missing, fall back to PATH.
    for name in ("7za.exe", "7za", "7z.exe", "7z"):
        candidate = tests_dir / name
        if candidate.exists():
            return candidate
    on_path = shutil.which("7za") or shutil.which("7z")
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        "7za.exe not found in tests/ or on PATH. Download 7-Zip console "
        "executable (7za.exe) from https://www.7-zip.org/download.html and "
        "place it in the tests/ directory."
    )


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def check(condition: bool, msg: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    return condition


def verify_zipfile_crc(archive: Path) -> bool:
    """Use Python zipfile to open and CRC-test all entries."""
    try:
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad is not None:
                print(f"  zipfile: first bad entry: {bad}")
                return False
            print(f"  zipfile: {len(zf.infolist())} entries, all CRCs OK")
            return True
    except zipfile.BadZipFile as e:
        print(f"  zipfile: BadZipFile: {e}")
        return False
    except Exception as e:
        print(f"  zipfile: error: {e}")
        return False


def verify_7za(archive: Path, seven_zip: Path) -> bool:
    r = run([str(seven_zip), "t", str(archive)])
    ok = r.returncode == 0 and "Everything is Ok" in r.stdout
    if not ok:
        print(f"  7za stdout: {r.stdout[-400:]}")
        print(f"  7za stderr: {r.stderr[-400:]}")
    return ok


def extract_and_compare(archive: Path, originals: list[Path],
                        workdir: Path, seven_zip: Path,
                        use_zipfile: bool) -> bool:
    """Extract via 7za (handles all methods) or zipfile (Store+Deflate only),
    then byte-compare against originals by basename."""
    extract_dir = workdir / "extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    if use_zipfile:
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            print(f"  zipfile extract error: {e}")
            return False
    else:
        r = run([str(seven_zip), "x", "-y", f"-o{extract_dir}", str(archive)])
        if r.returncode != 0:
            print(f"  7za x stdout: {r.stdout[-400:]}")
            print(f"  7za x stderr: {r.stderr[-400:]}")
            return False

    ok = True
    for orig in originals:
        # The entry name in the archive equals the basename.
        extracted = extract_dir / orig.name
        if not extracted.exists():
            print(f"  missing extracted file: {extracted.name}")
            ok = False
            continue
        h1 = sha256_file(orig)
        h2 = sha256_file(extracted)
        if not check(h1 == h2, f"sha256 match: {orig.name}"):
            ok = False
    return ok


def stage0_smoke(binary: Path) -> bool:
    print("[Stage 0] Smoke test: binary runs and --version works")
    if not binary.exists():
        return check(False, f"binary exists at {binary}")
    r = run([str(binary), "--version"])
    return check(r.returncode == 0, "exit code 0") and \
        check(bool(r.stdout.strip()), "prints a version string")


def make_small_files(workdir: Path, n: int, size: int = 32) -> list[Path]:
    d = workdir / "inputs"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    paths = []
    for i in range(n):
        p = d / f"file_{i:05d}.bin"
        p.write_bytes(struct.pack(">I", i) * (size // 4))
        paths.append(p)
    return paths


def write_test_zip(binary: Path, archive: Path, files: list[Path],
                   mode: str, extra: list[str] | None = None,
                   workdir: Path | None = None) -> bool:
    # Use a list file ('@listfile') when there are many files, to avoid
    # Windows' 32 KB command-line length limit.
    if len(files) > 64 and workdir is not None:
        list_file = workdir / "inputs.txt"
        list_file.write_text("\n".join(str(f) for f in files))
        cmd = [str(binary), mode, str(archive), f"@{list_file}"]
    else:
        cmd = [str(binary), mode, str(archive)] + [str(f) for f in files]
    if extra:
        cmd += extra
    r = run(cmd)
    if r.returncode != 0:
        print(f"  fzip stdout: {r.stdout}")
        print(f"  fzip stderr: {r.stderr}")
    return r.returncode == 0


def stage1_store(binary: Path, workdir: Path, seven_zip: Path) -> bool:
    print("[Stage 1] Store-method ZIP round-trip")
    files = make_small_files(workdir, 5, size=64)
    # Add a non-trivial text file.
    txt = workdir / "inputs" / "hello.txt"
    txt.write_text("Hello, fzip!\n" * 16)
    files.append(txt)
    archive = workdir / "stage1.zip"
    if not check(write_test_zip(binary, archive, files, "store", workdir=workdir),
                 "fzip store succeeds"):
        return False
    if not check(verify_zipfile_crc(archive), "zipfile CRC test"):
        return False
    if not check(verify_7za(archive, seven_zip), "7za t"):
        return False
    return check(extract_and_compare(archive, files, workdir, seven_zip,
                                     use_zipfile=True),
                 "extract + byte-compare")


def stage2_zip64(binary: Path, workdir: Path, seven_zip: Path) -> bool:
    print("[Stage 2] ZIP64 with >65535 entries")
    # 65540 tiny entries triggers ZIP64 by entry count (threshold is 65535).
    # Use 1-byte files to keep disk I/O and archive size small.
    files = make_small_files(workdir, 65540, size=1)
    archive = workdir / "stage2.zip"
    if not check(write_test_zip(binary, archive, files, "store", workdir=workdir),
                 "fzip store with 65540 entries succeeds"):
        return False
    # Python zipfile supports ZIP64. Use it for CRC.
    if not check(verify_zipfile_crc(archive), "zipfile CRC test (ZIP64)"):
        return False
    if not check(verify_7za(archive, seven_zip), "7za t (ZIP64)"):
        return False
    # Spot-check a few files instead of all 65540.
    sample = files[:8] + files[-8:]
    return check(extract_and_compare(archive, sample, workdir, seven_zip,
                                     use_zipfile=True),
                 "extract + byte-compare sample")


def stage3_deflate(binary: Path, workdir: Path, seven_zip: Path) -> bool:
    print("[Stage 3] Deflate-method ZIP round-trip")
    files = make_small_files(workdir, 4, size=4096)
    # Add a highly compressible text file.
    txt = workdir / "inputs" / "text.txt"
    txt.write_text("The quick brown fox jumps over the lazy dog.\n" * 256)
    files.append(txt)
    archive = workdir / "stage3.zip"
    if not check(write_test_zip(binary, archive, files, "deflate",
                                extra=["--level=6"], workdir=workdir),
                 "fzip deflate succeeds"):
        return False
    if not check(verify_zipfile_crc(archive), "zipfile CRC test (Deflate)"):
        return False
    if not check(verify_7za(archive, seven_zip), "7za t (Deflate)"):
        return False
    return check(extract_and_compare(archive, files, workdir, seven_zip,
                                     use_zipfile=True),
                 "extract + byte-compare")


def stage4_zstd(binary: Path, workdir: Path, seven_zip: Path) -> bool:
    print("[Stage 4] Zstd-method ZIP round-trip")
    files = make_small_files(workdir, 4, size=8192)
    txt = workdir / "inputs" / "text.txt"
    txt.write_text("The quick brown fox jumps over the lazy dog.\n" * 512)
    files.append(txt)
    archive = workdir / "stage4.zip"
    if not check(write_test_zip(binary, archive, files, "zstd",
                                extra=["--level=19"], workdir=workdir),
                 "fzip zstd succeeds"):
        return False
    # Python zipfile does NOT understand method 93; rely on 7za.
    if not check(verify_7za(archive, seven_zip), "7za t (Zstd method 93)"):
        return False
    return check(extract_and_compare(archive, files, workdir, seven_zip,
                                     use_zipfile=False),
                 "extract via 7za + byte-compare")


def stage5_auto(binary: Path, workdir: Path, seven_zip: Path) -> bool:
    print("[Stage 5] Auto codec selection round-trip")
    files = make_small_files(workdir, 4, size=4096)
    txt = workdir / "inputs" / "text.txt"
    txt.write_text("The quick brown fox jumps over the lazy dog.\n" * 256)
    files.append(txt)
    archive = workdir / "stage5.zip"
    if not check(write_test_zip(binary, archive, files, "auto",
                                workdir=workdir),
                 "fzip auto succeeds"):
        return False
    if not check(verify_7za(archive, seven_zip), "7za t (auto)"):
        return False
    return check(extract_and_compare(archive, files, workdir, seven_zip,
                                     use_zipfile=False),
                 "extract via 7za + byte-compare")


STAGES = {
    0: stage0_smoke,
    1: stage1_store,
    2: stage2_zip64,
    3: stage3_deflate,
    4: stage4_zstd,
    5: stage5_auto,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=int, required=True, choices=sorted(STAGES))
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--tests-dir", type=Path, default=None,
                    help="directory containing 7za.exe (default: tests/ next "
                         "to this script)")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)

    if args.tests_dir is None:
        args.tests_dir = Path(__file__).resolve().parent

    fn = STAGES[args.stage]

    if args.stage == 0:
        ok = fn(args.binary)
    else:
        try:
            seven_zip = find_7za(args.tests_dir)
        except FileNotFoundError as e:
            print(f"[FAIL] {e}")
            return 1
        ok = fn(args.binary, args.workdir, seven_zip)

    print(f"[{'PASS' if ok else 'FAIL'}] Stage {args.stage}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
