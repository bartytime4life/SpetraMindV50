#!/usr/bin/env python3
import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Optional deps only imported if enabled via env
TRACK_MLFLOW = os.getenv("WEIGHTS_TRACK_MLFLOW") == "1"
TRACK_WANDB = os.getenv("WEIGHTS_TRACK_WANDB") == "1"

if TRACK_MLFLOW:
    try:
        import mlflow
    except Exception:
        TRACK_MLFLOW = False
if TRACK_WANDB:
    try:
        import wandb
    except Exception:
        TRACK_WANDB = False

try:
    import yaml
except ImportError:
    print("Missing dependency: pyyaml", file=sys.stderr)
    sys.exit(2)

try:
    import jsonschema
except ImportError:
    print("Missing dependency: jsonschema", file=sys.stderr)
    sys.exit(2)

# -------- paths
ROOT = Path(__file__).resolve().parents[4]  # repo root (configs/.../weights -> up 4)
CONF_DIR = ROOT / "configs" / "symbolic" / "overrides" / "weights"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -------- logging (Hydra-safe: single root logger, no duplicate handlers)
logger = logging.getLogger("weights.validate")
logger.setLevel(logging.INFO)
if os.getenv("VERBOSE") == "1":
    logger.setLevel(logging.DEBUG)

if not logger.handlers:
    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # Rotating file
    fh = RotatingFileHandler(LOG_DIR / "weights_validate.log", maxBytes=2_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d | %(message)s"
        )
    )
    logger.addHandler(fh)

# JSONL event stream
EVENTS = LOG_DIR / "weights_events.jsonl"


def emit_event(kind: str, payload: dict):
    rec = {"ts": time.time(), "kind": kind, "payload": payload}
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def read_yaml(p: Path):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema():
    schema_path = CONF_DIR / "schema.json"
    if schema_path.suffix == ".yaml":
        return read_yaml(schema_path)
    return json.loads(schema_path.read_text())


def git_info():
    def _run(cmd):
        return subprocess.check_output(cmd, cwd=ROOT).decode("utf-8").strip()

    try:
        return {
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "dirty": bool(
                _run(["bash", "-lc", '[ -n "$(git status --porcelain)" ] && echo 1 || echo 0'])
            ),
            "remote": _run(["git", "remote", "get-url", "origin"]),
        }
    except Exception:
        return {"branch": None, "commit": None, "dirty": None, "remote": None}


def env_snapshot():
    keys = [
        "PYTHONPATH",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV",
        "WEIGHTS_TRACK_MLFLOW",
        "WEIGHTS_MLFLOW_EXPERIMENT",
        "WEIGHTS_MLFLOW_TRACKING_URI",
        "WEIGHTS_TRACK_WANDB",
        "WANDB_PROJECT",
    ]
    snap = {k: os.getenv(k) for k in keys}
    snap["python"] = sys.version
    snap["platform"] = platform.platform()
    return snap


def validate_one(path: Path, schema):
    doc = read_yaml(path)
    jsonschema.validate(instance=doc, schema=schema)
    # Apply constraints if set
    c = doc.get("constraints", {})
    w = doc.get("weights", {})
    violations = []
    mn, mx, clip = c.get("min"), c.get("max"), c.get("clip", False)
    for k, v in w.items():
        if mn is not None and v < mn:
            if clip:
                w[k] = mn
            violations.append((k, v, "min", mn))
        if mx is not None and v > mx:
            if clip:
                w[k] = mx
            violations.append((k, v, "max", mx))
    # Emit JSONL event
    emit_event(
        "validated",
        {
            "file": str(path.relative_to(ROOT)),
            "profile": doc.get("meta", {}).get("profile"),
            "version": doc.get("meta", {}).get("version"),
            "num_weights": len(w),
            "violations": violations,
        },
    )
    logger.info(
        "\u2713 %s (profile=%s, weights=%d, violations=%d)",
        path.name,
        doc.get("meta", {}).get("profile"),
        len(w),
        len(violations),
    )
    return doc


def maybe_track_ml(path: Path, summary: dict):
    if TRACK_MLFLOW:
        mlflow.set_tracking_uri(os.getenv("WEIGHTS_MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(os.getenv("WEIGHTS_MLFLOW_EXPERIMENT", "weights-overrides"))
        with mlflow.start_run(run_name=summary["profile"]):
            mlflow.log_params(
                {
                    "file": str(path),
                    "version": summary["version"],
                    "num_weights": summary["num_weights"],
                }
            )
            mlflow.log_dict(summary, "summary.json")
    if TRACK_WANDB:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "weights-overrides"),
            name=summary["profile"],
            reinit=True,
        )
        wandb.config.update(summary)
        run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, help="Validate only this YAML file")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)

    schema = load_schema()
    emit_event("run_start", {"git": git_info(), "env": env_snapshot()})

    targets = []
    if args.file:
        targets = [Path(args.file)]
    else:
        targets = [p for p in CONF_DIR.glob("*.yaml") if p.name != "schema.json"]

    ok = True
    for p in targets:
        try:
            doc = validate_one(p, schema)
            summary = {
                "profile": doc.get("meta", {}).get("profile"),
                "version": doc.get("meta", {}).get("version"),
                "file": str(p.relative_to(Path.cwd())),
                "num_weights": len(doc.get("weights", {})),
            }
            maybe_track_ml(p, summary)
        except Exception as e:
            ok = False
            logger.exception("Validation failed for %s", p)
            emit_event("validation_error", {"file": str(p), "error": str(e)})

    emit_event("run_end", {"ok": ok})
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
