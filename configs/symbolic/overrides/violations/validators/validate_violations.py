#!/usr/bin/env python3
"""SpectraMind V50 — Violations Config Validator."""

import argparse
import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from glob import glob

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required. pip install pyyaml", file=sys.stderr)
    sys.exit(2)

try:
    from jsonschema import Draft202012Validator, RefResolver
except ImportError:
    print("ERROR: jsonschema is required. pip install jsonschema", file=sys.stderr)
    sys.exit(2)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCHEMA_DIR = os.path.join(ROOT, "_schemas")
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

EVENT_STREAM = os.path.join(LOG_DIR, "violations_events.jsonl")
ROTATING_LOG = os.path.join(LOG_DIR, "violations_validator.log")


def setup_logging():
    logger = logging.getLogger("violations_validator")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(ROTATING_LOG, maxBytes=10_485_760, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


LOGGER = setup_logging()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_event(event: dict):
    event.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    with open(EVENT_STREAM, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def schema_path(name: str) -> str:
    return os.path.join(SCHEMA_DIR, name)


def compile_schema(name: str) -> Draft202012Validator:
    path = schema_path(name)
    with open(path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    store = {}
    for fname in ("violations.schema.yaml", "rule.schema.yaml", "context_override.schema.yaml"):
        with open(schema_path(fname), "r", encoding="utf-8") as sf:
            store[f"file://{SCHEMA_DIR}/{fname}"] = yaml.safe_load(sf)
    resolver = RefResolver(base_uri=f"file://{SCHEMA_DIR}/", referrer=schema, store=store)
    return Draft202012Validator(schema, resolver=resolver)


def validate_file(path, validator, kind):
    data = load_yaml(path)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        for e in errors:
            LOGGER.error("Invalid %s file=%s error=%s path=%s", kind, path, e.message, list(e.path))
        write_event({"kind": kind, "file": path, "status": "invalid", "errors": [e.message for e in errors]})
        return False
    else:
        LOGGER.info("Valid %s: %s", kind, path)
        write_event({"kind": kind, "file": path, "status": "valid"})
        return True


def main():
    ap = argparse.ArgumentParser(description="Validate SpectraMind V50 violations configs")
    ap.add_argument("--file", type=str, help="Single file to validate (auto-detects schema by path)")
    ap.add_argument("--all", action="store_true", help="Validate base, rules, and contexts")
    args = ap.parse_args()

    violations_schema = compile_schema("violations.schema.yaml")
    rule_schema = compile_schema("rule.schema.yaml")
    context_schema = compile_schema("context_override.schema.yaml")

    ok = True

    def validate_auto(path):
        nonlocal ok
        if "/rules/" in path:
            ok = validate_file(path, rule_schema, "rule") and ok
        elif "/contexts/" in path:
            ok = validate_file(path, context_schema, "context_override") and ok
        else:
            ok = validate_file(path, violations_schema, "violations") and ok

    if args.file and not args.all:
        validate_auto(os.path.abspath(args.file))
    else:
        targets = []
        base = os.path.join(ROOT, "base.yaml")
        targets.append(base)
        targets.extend(sorted(glob(os.path.join(ROOT, "rules", "*.yaml"))))
        for bucket in ("competition", "instruments", "profiles", "events"):
            targets.extend(sorted(glob(os.path.join(ROOT, "contexts", bucket, "*.yaml"))))
        for t in targets:
            validate_auto(t)

    if not ok:
        sys.exit(1)
    LOGGER.info("All validations passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
