#!/usr/bin/env python3
# Quick local validator using jsonschema (pip install jsonschema pyyaml)
import json
import pathlib
import sys

import yaml
from jsonschema import Draft7Validator, validate

ROOT = pathlib.Path(__file__).resolve().parent
SCHEMAS = {
    "events": ROOT / "_schemas" / "event_override.schema.json",
    "competition": ROOT / "_schemas" / "competition_override.schema.json",
    "molecules": ROOT / "_schemas" / "molecule_override.schema.json",
}


def load_json(p):
    return json.loads(pathlib.Path(p).read_text())


def load_yaml(p):
    return yaml.safe_load(pathlib.Path(p).read_text())


def run_validate(schema_path, doc_path):
    schema = load_json(schema_path)
    doc = load_json(doc_path) if str(doc_path).endswith(".json") else load_yaml(doc_path)
    Draft7Validator.check_schema(schema)
    validate(instance=doc, schema=schema)
    print(f"OK: {doc_path}")


def main():
    targets = [
        (SCHEMAS["events"], ROOT / "events" / "example_transit_event.yaml"),
        (SCHEMAS["events"], ROOT / "events" / "example_transit_event.json"),
        (SCHEMAS["events"], ROOT / "events" / "example_anomaly_recovery.yaml"),
        (SCHEMAS["competition"], ROOT / "competition" / "strict_competition.yaml"),
        (SCHEMAS["competition"], ROOT / "competition" / "balanced_competition.yaml"),
        (SCHEMAS["competition"], ROOT / "competition" / "strict_competition.json"),
        (SCHEMAS["molecules"], ROOT / "molecules" / "h2o_focus.yaml"),
        (SCHEMAS["molecules"], ROOT / "molecules" / "co2_focus.yaml"),
        (SCHEMAS["molecules"], ROOT / "molecules" / "ch4_focus.yaml"),
        (SCHEMAS["molecules"], ROOT / "molecules" / "h2o_focus.json"),
    ]
    try:
        for schema, doc in targets:
            run_validate(schema, doc)
        print("All validations passed.")
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
