#!/usr/bin/env python3
"""
Local validator for instrument profiles and override bundles.
Requires: pip install jsonschema pyyaml
"""

import json
import pathlib
import sys

import yaml
from jsonschema import Draft7Validator, validate

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCH_INST = ROOT / "configs/symbolic/instruments/instrument.schema.yaml"
SCH_OVR = ROOT / "configs/symbolic/overrides/_schemas/instrument_override.schema.json"


def load_yaml(p):
    return yaml.safe_load(pathlib.Path(p).read_text())


def load_json(p):
    return json.loads(pathlib.Path(p).read_text())


def check_schema(schema_path, doc_path):
    schema = (
        load_yaml(schema_path)
        if schema_path.suffix in (".yaml", ".yml")
        else load_json(schema_path)
    )
    Draft7Validator.check_schema(schema)
    doc = load_yaml(doc_path) if doc_path.suffix in (".yaml", ".yml") else load_json(doc_path)
    validate(instance=doc, schema=schema)
    print(f"OK: {doc_path}")


def main():
    # Instrument profiles
    profs = [
        ROOT / "configs/symbolic/instruments/ariel/airs.yaml",
        ROOT / "configs/symbolic/instruments/ariel/fgs.yaml",
        ROOT / "configs/symbolic/instruments/jwst/nirspec.yaml",
        ROOT / "configs/symbolic/instruments/simulated/ideal.yaml",
    ]
    # Override bundles
    ovrs = [
        ROOT / "configs/symbolic/overrides/instruments/dev-ariel.yaml",
        ROOT / "configs/symbolic/overrides/instruments/jwst-benchmark.yaml",
        ROOT / "configs/symbolic/overrides/instruments/fast-sim.yaml",
        ROOT / "configs/symbolic/overrides/instruments/jitter-stress.yaml",
        ROOT / "configs/symbolic/overrides/instruments/thermal-drift.yaml",
        ROOT / "configs/symbolic/overrides/instruments/recovery-safe.yaml",
        ROOT / "configs/symbolic/overrides/instruments/qa-battery.yaml",
        ROOT / "configs/symbolic/overrides/instruments/fgs-guiding.yaml",
    ]
    try:
        for p in profs:
            check_schema(SCH_INST, p)
        for o in ovrs:
            check_schema(SCH_OVR, o)
        print("All validations passed.")
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
