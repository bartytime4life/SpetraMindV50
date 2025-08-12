#!/usr/bin/env python3
# --- sync_dropins.py (full) ---
from __future__ import annotations
import argparse, datetime as dt, difflib, hashlib, os, shutil, sys
from pathlib import Path
from typing import Tuple

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def is_text_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            sample = f.read(8192)
        sample.decode("utf-8")
        return True
    except Exception:
        return False

def rel_paths(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def unified_diff(a_path: Path, b_path: Path) -> str:
    a_text = load_text(a_path) if a_path.exists() else ""
    b_text = load_text(b_path)
    a_lines = a_text.splitlines(keepends=True)
    b_lines = b_text.splitlines(keepends=True)
    diff = difflib.unified_diff(
        a_lines, b_lines,
        fromfile=str(a_path), tofile=str(b_path),
        lineterm="", n=3,
    )
    return "".join(diff)

def copy_with_backup(dst: Path, src: Path, backup_dir: Path | None, ts: str) -> Path | None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    bak_path = None
    if dst.exists():
        if backup_dir is None:
            backup_dir = dst.parent / ".backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        bak_path = backup_dir / f"{dst.name}.{ts}.bak"
        shutil.copy2(dst, bak_path)
    shutil.copy2(src, dst)
    return bak_path

def main() -> int:
    ap = argparse.ArgumentParser(description="Sync drop-in files into a repo (with backups).")
    ap.add_argument("--repo", required=True, help="Path to the repository root (target).")
    ap.add_argument("--dropins", required=True, help="Path to new files (same relative layout).")
    ap.add_argument("--apply", action="store_true", help="Apply changes (make backups, then replace).")
    ap.add_argument("--yes", action="store_true", help="Skip interactive prompts when applying.")
    ap.add_argument("--backup-root", default=None, help="Optional root for backups. Defaults near destination.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    drop = Path(args.dropins).resolve()
    if not repo.exists() or not repo.is_dir():
        print(f"[ERR] --repo not found: {repo}", file=sys.stderr)
        return 2
    if not drop.exists() or not drop.is_dir():
        print(f"[ERR] --dropins not found: {drop}", file=sys.stderr)
        return 2

    backup_root = Path(args.backup_root).resolve() if args.backup_root else None
    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    drop_files = rel_paths(drop)
    if not drop_files:
        print(f"[WARN] No files found under {drop}")
        return 0

    missing, identical, different = [], [], []
    text_diffs: list[Tuple[Path, str]] = []
    bin_changes: list[Path] = []

    for src in drop_files:
        rel = src.relative_to(drop)
        dst = repo / rel

        if not dst.exists():
            missing.append(rel)
            continue

        try:
            if is_text_file(src) and is_text_file(dst):
                if load_text(src) == load_text(dst):
                    identical.append(rel)
                else:
                    different.append(rel)
                    text_diffs.append((rel, unified_diff(dst, src)))
            else:
                if sha256(src) == sha256(dst):
                    identical.append(rel)
                else:
                    different.append(rel)
                    bin_changes.append(rel)
        except Exception as e:
            print(f"[ERR] Failed to compare {rel}: {e}", file=sys.stderr)

    print("\n=== Drop-in Sync Report ===")
    print(f"Repo:     {repo}")
    print(f"Drop-ins: {drop}\n")
    print(f"IDENTICAL: {len(identical)}")
    print(f"MISSING:  {len(missing)}")
    print(f"DIFFERS:  {len(different)}")
    if missing:
        print("\n-- Missing in repo (will be created if --apply):")
        for r in sorted(missing):
            print(f"  + {r}")
    if different:
        print("\n-- Differing files:")
        for r in sorted(different):
            print(f"  ~ {r}")

    if text_diffs:
        print("\n=== Unified Diffs (repo \u2190 drop-in) ===")
        for rel, diff in text_diffs:
            print(f"\n--- {rel} ---")
            print(diff if diff.strip() else "(no textual diff?)")

    if bin_changes:
        print("\n=== Binary/Non-UTF8 differences (hash only) ===")
        for rel in sorted(bin_changes):
            print(f"  * {rel}")

    if args.apply:
        if not args.yes:
            resp = input("\nApply changes? This will create .bak files then replace. [y/N]: ").strip().lower()
            if resp not in {"y", "yes"}:
                print("Aborted."); return 1
        changed_count = 0
        to_write = sorted(set(missing) | set(different))
        for rel in to_write:
            src = drop / rel
            dst = repo / rel
            bak = copy_with_backup(dst, src, backup_root, ts)
            if bak:
                try:
                    print(f"[bak] {rel} -> {bak.relative_to(repo)}")
                except Exception:
                    print(f"[bak] {rel} -> {bak}")
            print(f"[write] {rel}")
            changed_count += 1
        print(f"\nDone. {changed_count} file(s) written.")
        print("Tip: review via `git status` and `git diff`, then commit:")
        print("  git add -A && git commit -m 'Sync drop-ins (with backups)'")
        return 0

    # exit code useful for CI: 0 if no changes needed, 3 if differences exist
    return 0 if (not missing and not different) else 3

if __name__ == "__main__":
    sys.exit(main())
