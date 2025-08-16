from __future__ import annotations

import argparse
import sys
from typing import List

from .schema_registry import export_json_schemas, list_registered_models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spectramind.reporting.schema.export",
        description="Export JSON Schemas for SpectraMind reporting models.",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        help="Output directory for *.schema.json files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered models and exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name in list_registered_models():
            print(name)
        return 0

    written: List[str] = export_json_schemas(args.out_dir)
    for p in written:
        print(f"WROTE {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
