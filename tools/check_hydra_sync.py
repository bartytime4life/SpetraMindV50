#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Hydra Trees Sync Checker

## Purpose

Ensures that configs/hydra/** and conf/hydra/** are exact mirrors for:

* File set (same relative paths)
* File content (byte-for-byte by default; optional normalized comparisons)

## Features

* Pretty summarized report with per-file status
* Exit non-zero on divergence (CI-friendly)
* Ignorable patterns (e.g., README.md) via --ignore-glob
* Optional normalization (strip trailing spaces, normalize newlines)
* --fix mode to automatically copy changes from configs/hydra -> conf/hydra

## Usage

# Dry check (local)

python tools/check_hydra_sync.py

# CI mode with verbose output

python tools/check_hydra_sync.py --ci --verbose

# Auto-fix conf/hydra to match configs/hydra

python tools/check_hydra_sync.py --fix

## Notes

* Source of truth is configs/hydra. The checker compares/replicates into conf/hydra.
* To keep behavior deterministic, we DO NOT delete extra files in conf/ unless --prune is specified.

## Return Codes

0 = in sync / fixed successfully
1 = divergence detected (and not fixed), or unrecoverable error

## Author

SpectraMind V50 · Master Architect
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import shutil
import sys
from typing import Dict, List, Tuple

SRC = os.path.normpath("configs/hydra")
DST = os.path.normpath("conf/hydra")

DEFAULT_IGNORES = [
    # Add ignorable patterns here if needed:
    # "README.md",
]


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def info(msg: str) -> None:
    print(f"[INFO]  {msg}")


def warn(msg: str) -> None:
    print(f"[WARN]  {msg}")


def is_ignored(rel_path: str, ignore_globs: List[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pat) for pat in ignore_globs)


def list_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, root)
            files.append(rel.replace("\\", "/"))
    files.sort()
    return files


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def normalize_bytes(b: bytes) -> bytes:
    # Normalize newlines to \n and strip trailing spaces on each line
    s = b.decode("utf-8", errors="replace")
    lines = s.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip(" \t") for ln in lines]
    return ("\n".join(lines)).encode("utf-8")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def compare_file(
    src_path: str, dst_path: str, normalize: bool
) -> Tuple[bool, str, str]:
    a = read_bytes(src_path)
    b = read_bytes(dst_path) if os.path.exists(dst_path) else b""
    if normalize:
        a = normalize_bytes(a)
        b = normalize_bytes(b)
    return (a == b, sha256(a), sha256(b))


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def copy_file(src_path: str, dst_path: str) -> None:
    ensure_dir(dst_path)
    shutil.copy2(src_path, dst_path)


def prune_extras(
    dst_root: str,
    expected_rel_files: List[str],
    ignore_globs: List[str],
    verbose: bool,
) -> List[str]:
    removed = []
    actual = list_files(dst_root)
    expected_set = set(expected_rel_files)
    for rel in actual:
        if is_ignored(rel, ignore_globs):
            continue
        if rel not in expected_set:
            full = os.path.join(dst_root, rel)
            if verbose:
                info(f"Pruning extra: {rel}")
            os.remove(full)
            removed.append(rel)
            # Remove empty directories upward
            d = os.path.dirname(full)
            while d and d != dst_root and os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
                d = os.path.dirname(d)
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description="Check/Sync configs/hydra <-> conf/hydra")
    ap.add_argument("--ci", action="store_true", help="CI mode (fail hard on issues)")
    ap.add_argument("--verbose", action="store_true", help="Verbose reporting")
    ap.add_argument(
        "--ignore-glob", action="append", default=[], help="Glob to ignore (repeatable)"
    )
    ap.add_argument(
        "--normalize", action="store_true", help="Normalize text before comparing"
    )
    ap.add_argument(
        "--fix",
        action="store_true",
        help="Auto-copy missing/changed files from configs to conf",
    )
    ap.add_argument(
        "--prune",
        action="store_true",
        help="Delete extra files found only in conf/hydra",
    )
    args = ap.parse_args()

    if not os.path.isdir(SRC):
        die(f"Missing source directory: {SRC}")
    if not os.path.isdir(DST):
        warn(f"Missing destination directory: {DST}; creating it")
        os.makedirs(DST, exist_ok=True)

    ignore_globs = list(DEFAULT_IGNORES) + list(args.ignore_glob)
    if args.verbose and ignore_globs:
        info(f"Ignore patterns: {ignore_globs}")

    src_files = [f for f in list_files(SRC) if not is_ignored(f, ignore_globs)]
    dst_files = [f for f in list_files(DST) if not is_ignored(f, ignore_globs)]

    src_set = set(src_files)
    dst_set = set(dst_files)

    only_in_src = sorted(src_set - dst_set)
    only_in_dst = sorted(dst_set - src_set)

    # Compare shared files
    changed = []
    same = []
    diffs: Dict[str, Tuple[str, str]] = {}

    for rel in sorted(src_set & dst_set):
        src_path = os.path.join(SRC, rel)
        dst_path = os.path.join(DST, rel)
        eq, hsrc, hdst = compare_file(src_path, dst_path, args.normalize)
        if eq:
            same.append(rel)
        else:
            changed.append(rel)
            diffs[rel] = (hsrc, hdst)

    # Report
    print("\n=== Hydra Trees Sync Report ===")
    print(f"Source:      {SRC}")
    print(f"Destination: {DST}")
    print(f"Normalize:   {args.normalize}")
    print(f"CI Mode:     {args.ci}")
    print("-------------------------------")

    if only_in_src:
        print("\nFiles only in source (need copy):")
        for rel in only_in_src:
            print(f"  + {rel}")

    if only_in_dst:
        print("\nFiles only in destination (extra files):")
        for rel in only_in_dst:
            print(f"  - {rel}")

    if changed:
        print("\nFiles with content differences:")
        for rel in changed:
            hsrc, hdst = diffs[rel]
            print(f"  * {rel}")
            if args.verbose:
                print(f"      src={hsrc}")
                print(f"      dst={hdst}")

    if same and args.verbose:
        print("\nFiles in sync:")
        for rel in same:
            print(f"    = {rel}")

    # Auto-fix path (copy and optionally prune)
    any_issue = bool(only_in_src or only_in_dst or changed)
    if args.fix:
        # Copy missing/changed from SRC -> DST
        for rel in only_in_src + changed:
            src_path = os.path.join(SRC, rel)
            dst_path = os.path.join(DST, rel)
            info(f"Syncing {rel}")
            copy_file(src_path, dst_path)
        # Prune extras from DST
        if args.prune and only_in_dst:
            prune_extras(DST, src_files, ignore_globs, args.verbose)
        # Recompute to confirm clean state
        src_files2 = [f for f in list_files(SRC) if not is_ignored(f, ignore_globs)]
        dst_files2 = [f for f in list_files(DST) if not is_ignored(f, ignore_globs)]
        src_set2, dst_set2 = set(src_files2), set(dst_files2)
        post_only_src = sorted(src_set2 - dst_set2)
        post_only_dst = sorted(dst_set2 - src_set2)
        post_changed = []
        for rel in sorted(src_set2 & dst_set2):
            src_path = os.path.join(SRC, rel)
            dst_path = os.path.join(DST, rel)
            eq, _, _ = compare_file(src_path, dst_path, args.normalize)
            if not eq:
                post_changed.append(rel)

        fixed = not (post_only_src or post_only_dst or post_changed)
        print("\n=== Post-Fix Verification ===")
        if fixed:
            print("Hydra trees are now in sync ✅")
            sys.exit(0)
        else:
            if post_only_src:
                print("Still only in source:", post_only_src)
            if post_only_dst:
                print("Still only in destination:", post_only_dst)
            if post_changed:
                print("Still changed:", post_changed)
            die("Hydra trees still out of sync after --fix", code=1)

    # No fix: decide exit code
    if any_issue:
        msg = "Divergence detected between configs/hydra and conf/hydra"
        if args.ci:
            die(msg, code=1)
        else:
            warn(msg)
            sys.exit(1)
    else:
        print("\nHydra trees are in sync ✅")
        sys.exit(0)


if __name__ == "__main__":
    main()
