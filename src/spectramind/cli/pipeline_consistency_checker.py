from __future__ import annotations

from pathlib import Path
from rich import print

CHECKS = [
    ("Root CLI present", Path("spectramind.py").exists()),
    ("Configs exist", any(Path("configs").rglob("*.yaml"))),
    ("Diagnostics writer present", Path("src/spectramind/diagnostics/generate_html_report.py").exists()),
]


def main() -> int:
    failed = [name for name, ok in CHECKS if not ok]
    if failed:
        for name in failed:
            print(f"[red]❌ {name}")
        return 1
    print("[green]✅ Pipeline consistency ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
