#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Schema Validator for Calibration Configs

Validates YAML files against the JSON Schemas in this directory.

Usage:
python configs/calibration/schema/validate.py --files configs/calibration/*.yaml
python configs/calibration/schema/validate.py --dir configs/calibration

Exit codes:
0 = all files valid
1 = at least one file invalid or runtime error

Notes:
• Draft 2020-12 with $ref composition across local schema files.
• Prints a concise error table and a rich per-file report on failure.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import yaml  # PyYAML
except Exception as e:
    print("ERROR: PyYAML is required. pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    import jsonschema
    from jsonschema.validators import Draft202012Validator
    from jsonschema import RefResolver
except Exception as e:
    print("ERROR: jsonschema >=4 is required. pip install jsonschema", file=sys.stderr)
    sys.exit(1)

SCHEMA_FILES = [
    "common_defs.schema.json",
    "frame_corrections.schema.json",
    "cosmic_ray.schema.json",
    "photometry_extraction.schema.json",
    "trace_extraction.schema.json",
    "normalization_alignment.schema.json",
    "uncertainty_calibration.schema.json",
    "validation_checks.schema.json",
    "products_manifest.schema.json",
    "calibration_pipeline.schema.json",
]

def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_schemas(schema_dir: Path):
    store = {}
    for name in SCHEMA_FILES:
        p = schema_dir / name
        if not p.exists():
            print(f"ERROR: Missing schema file: {p}", file=sys.stderr)
            sys.exit(1)
        data = _load_json(p)
        store[data.get("$id", f"file://{p.resolve()}")] = data
        store[name] = data
    return store

def make_resolver(schema_dir: Path, store: dict) -> RefResolver:
    base_uri = f"file://{schema_dir.resolve()}/"
    return RefResolver(base_uri=base_uri, referrer=None, store=store)

def validate_file(cfg_path: Path, pipeline_schema: dict, resolver: RefResolver) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    validator = Draft202012Validator(schema=pipeline_schema, resolver=resolver)
    try:
        data = _load_yaml(cfg_path)
    except Exception as e:
        return False, [f"YAML parse error: {e}"]
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        loc = "/".join([str(x) for x in err.path]) or ""
        errors.append(f"{cfg_path.name}: {loc}: {err.message}")
    return (len(errors) == 0), errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", help="YAML files or globs to validate")
    ap.add_argument("--dir", default=None, help="Directory to scan for *.yaml")
    ap.add_argument("--schema-dir", default=None, help="Schema directory (default: this file's dir)")
    args = ap.parse_args()

    schema_dir = Path(args.schema_dir) if args.schema_dir else Path(__file__).parent
    store = load_schemas(schema_dir)
    resolver = make_resolver(schema_dir, store)
    pipeline_schema = store.get("calibration_pipeline.schema.json") or store.get(
        "https://spectramind.v50/schema/calibration/calibration_pipeline.schema.json"
    )
    if not pipeline_schema:
        print("ERROR: calibration_pipeline.schema.json not loaded", file=sys.stderr)
        sys.exit(1)

    files: List[str] = []
    if args.files:
        for pattern in args.files:
            files.extend(glob.glob(pattern))
    if args.dir:
        files.extend([str(p) for p in Path(args.dir).glob("*.yaml")])

    files = sorted(set(files))
    if not files:
        print("No files to validate. Provide --files or --dir", file=sys.stderr)
        sys.exit(1)

    any_fail = False
    all_errors: List[str] = []
    for f in files:
        ok, errs = validate_file(Path(f), pipeline_schema, resolver)
        if not ok:
            any_fail = True
            all_errors.extend(errs)

    if any_fail:
        print("VALIDATION FAILED:")
        for e in all_errors:
            print(f"  - {e}")
        print(f"\nChecked {len(files)} file(s); {len(all_errors)} error(s) found.")
        sys.exit(1)
    else:
        print(f"OK: {len(files)} file(s) valid.")
        sys.exit(0)

if __name__ == "__main__":
    main()
