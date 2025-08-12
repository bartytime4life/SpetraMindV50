from __future__ import annotations

from pathlib import Path
from rich import print

REQUIRED = [
    "configs/config_v50.yaml",
    "spectramind.py",
    "src/spectramind/diagnostics/generate_html_report.py",
]


def main() -> int:
    missing = [p for p in REQUIRED if not Path(p).exists()]
    if missing:
        print(f"[red]❌ Missing: {missing}")
        return 1
    print("[green]✅ Selftest (module) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
