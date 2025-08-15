#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Repo Auditor & Scaffolder — SpectraMind V50 (Ariel Data Challenge 2025)
Author: ChatGPT (GPT-5 Thinking) — “master coder & architect” protocols

What this does:

1. Scans your working directory (the repository root) against a curated manifest
   of expected files and directories for SpectraMind V50 (with emphasis on
   configs/symbolic/overrides/* and nearby config systems you've asked for).
2. Produces:

   * Console summary of missing dirs/files
   * Rotating file logs: logs/repo_audit.log
   * JSONL event stream: reports/repo_audit_events.jsonl
   * JSON report: reports/repo_audit_report.json
3. With --apply:

   * Creates any missing directories
   * Writes structured starter files with Hydra-safe YAML, schemas, and
     commented templates (“batteries included”)
4. With --git:

   * Creates a branch (default: chore/repo-audit-<timestamp>)
   * git add / commit / push (origin) with a clean message
5. Captures:

   * Git status / branch / origin
   * Python & package snapshot
   * ENV snapshot (redacted safe keys)
6. Optional telemetry toggles (off by default; enable via env):

   * Enable MLflow:   SMV50_MLFLOW_ENABLE=1
   * Enable Weights&Biases: SMV50_WANDB_ENABLE=1

Usage:
Dry-run audit (recommended first):
python tools/repo_audit_and_scaffold.py --repo .

Apply scaffolding (create missing), commit and push on a new branch:
python tools/repo_audit_and_scaffold.py --repo . --apply --git

Notes:

* Idempotent: rerunning won't duplicate content; it will skip existing files.
* All new files include clear headers and TODOs, with minimal-yet-complete,
  production-ready defaults consistent with Hydra & CI.
"""

import argparse
import dataclasses
import datetime as _dt
import json
import json as _json
import logging
import logging.handlers
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ENV_VALUE_TRUNCATE = 200
MAX_PRESENT_ITEMS = 50

# -----------------------------

# Logging (console + rotating)

# -----------------------------


def _setup_logging(repo_root: Path, verbose: bool) -> logging.Logger:
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "repo_audit.log"

    logger = logging.getLogger("repo_audit")
    logger.setLevel(logging.DEBUG)

    # File handler: rotating, no ANSI, timestamped
    fh = logging.handlers.RotatingFileHandler(
        str(log_path), maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    ffmt = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    fh.setFormatter(ffmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    cfmt = logging.Formatter(fmt="%(message)s")
    ch.setFormatter(cfmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# -----------------------------

# JSONL event stream

# -----------------------------


class JSONLEmitter:
    def __init__(self, out_path: Path):
        self.out_path = out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(out_path, "a", encoding="utf-8")

    def emit(self, event: dict[str, Any]):
        event = dict(event)
        event.setdefault("ts", _dt.datetime.utcnow().isoformat() + "Z")
        self._f.write(_json.dumps(event, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


# -----------------------------

# Helpers

# -----------------------------


def _now_utc() -> str:
    return _dt.datetime.utcnow().isoformat() + "Z"


def _read_git(repo_root: Path) -> dict[str, Any]:
    def run(cmd: list[str]) -> tuple[int, str]:
        try:
            out = subprocess.check_output(
                cmd, cwd=str(repo_root), stderr=subprocess.STDOUT, text=True
            )
            return 0, out.strip()
        except subprocess.CalledProcessError as e:
            return e.returncode, e.output.strip()

    info = {}
    rc, b = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    info["branch"] = b if rc == 0 else None
    rc, s = run(["git", "status", "--porcelain=1", "--branch"])
    info["status"] = s if rc == 0 else None
    rc, r = run(["git", "remote", "-v"])
    info["remotes"] = r if rc == 0 else None
    rc, h = run(["git", "rev-parse", "HEAD"])
    info["head"] = h if rc == 0 else None
    return info


def _safe_env_snapshot() -> dict[str, Any]:
    # Redact common secrets; keep helpful build info
    redacted_keys = re.compile(
        r"(TOKEN|SECRET|KEY|PASSWORD|PASS|AWS*|GCP_|AZURE_|OPENAI_|HUGGINGFACE_|WANDB_|MLFLOW_)",
        re.I,
    )
    snap = {}
    for k, v in os.environ.items():
        if redacted_keys.search(k):
            snap[k] = "<redacted>"
        else:
            value = v
            if len(value) > ENV_VALUE_TRUNCATE:
                value = value[:ENV_VALUE_TRUNCATE] + "…"
            snap[k] = value
    return snap


def _python_snapshot() -> dict[str, Any]:
    py = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    # Try to collect `pip freeze`
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        py["pip_freeze"] = lines[:5000]  # truncate if massive
    except Exception as e:
        py["pip_freeze_error"] = str(e)
    return py


# -----------------------------

# Manifest (curated expectations)

# -----------------------------

# Each entry: path -> dict(type, template, description)

# type: "dir" or "file"

# template: str or None (file content if created)

# We keep content compact but production-ready, with comments and TODOs.

MANIFEST: dict[str, dict[str, Any]] = {}


def _add_dir(path: str, description: str):
    MANIFEST[path] = {"type": "dir", "template": None, "description": description}


def _add_file(path: str, description: str, template: str):
    MANIFEST[path] = {"type": "file", "template": template, "description": description}


# ---- Core symbolic overrides focus (per your recent requests) ----

_add_dir(
    "configs/symbolic/overrides",
    "Root of symbolic override packs (profiles, weights, violations, events, instruments, competition).",
)
_add_dir("configs/symbolic/overrides/_schemas", "JSON/YAML schemas for validating overrides.")
_add_dir(
    "configs/symbolic/overrides/profiles",
    "Symbolic profile packs (selects logical bundles across regions/molecules/instruments).",
)
_add_dir("configs/symbolic/overrides/weights", "Override weights for symbolic losses / rules.")
_add_dir(
    "configs/symbolic/overrides/violations",
    "Explicit rule violation templates / expected anomaly patterns.",
)
_add_dir(
    "configs/symbolic/overrides/events", "Event-level overrides (e.g., per-planet, per-transit)."
)
_add_dir(
    "configs/symbolic/overrides/instruments", "Instrument-specific overrides (AIRS, FGS1, etc.)."
)
_add_dir(
    "configs/symbolic/overrides/competition",
    "Competition-mode overrides for NeurIPS submission constraints.",
)

# Schemas (minimal but real, Hydra-safe)

_add_file(
    "configs/symbolic/overrides/_schemas/override_schema.yaml",
    "Hydra-safe schema for override files.",
    """# SpectraMind V50 — Override Schema (Hydra-safe)

# Purpose: Validate symbolic override files via jsonschema/yamale/cerberus (your choice).

# Fields are optional by default; strict mode can be enabled in CLI validators.

version: "1.0"
type: "object"
properties:
  name: { type: "string" }
  description: { type: "string" }
  applies_to:
    type: "object"
    properties:
      instruments: { type: "array", items: { type: "string" } }
      molecules:   { type: "array", items: { type: "string" } }
      regions:     { type: "array", items: { type: "string" } }
      planets:     { type: "array", items: { type: "string" } }
  rules:
    type: "array"
    items:
      type: "object"
      properties:
        id:          { type: "string" }
        weight:      { type: ["number", "string"] }   # allow ${oc.env:…} Hydra resolvers
        mode:        { type: "string", enum: ["soft", "hard"] }
        parameters:  { type: "object" }
      required: []
""",
)

# Profiles

_add_file(
    "configs/symbolic/overrides/profiles/base.yaml",
    "Base symbolic profiles (mergeable).",
    """# SpectraMind V50 — Symbolic Profiles (Base)

# Mergeable via Hydra defaults/layers. Override per-planet / per-cluster as needed.

profiles:
  default:
    description: "Balanced rules for AIRS + FGS1"
    applies_to:
      instruments: ["AIRS", "FGS1"]
    rules:
      - id: "smoothness_l2"
        weight: 1.0
        mode: "soft"
        parameters: { window: 3 }
      - id: "nonnegativity"
        weight: 1.0
        mode: "hard"
        parameters: {}
      - id: "molecular_coherence"
        weight: 0.5
        mode: "soft"
        parameters: { tolerance: 0.15 }
  kaggle_fast:
    description: "Runtime-constrained symbolic set"
    applies_to:
      instruments: ["AIRS"]
    rules:
      - id: "smoothness_l2"
        weight: 0.6
        mode: "soft"
        parameters: { window: 3 }
""",
)

# Weights

_add_file(
    "configs/symbolic/overrides/weights/base.yaml",
    "Default override weights (can be layered).",
    """# SpectraMind V50 — Override Weights (Base)
weights:
  smoothness_l2: 1.0
  nonnegativity: 1.0
  molecular_coherence: 0.5
  fft_asymmetry: 0.2
  photonic_alignment: 0.3
""",
)

# Violations

_add_file(
    "configs/symbolic/overrides/violations/anomaly_templates.yaml",
    "Known anomaly/violation patterns to stress-test the symbolic stack.",
    """# SpectraMind V50 — Violation Templates
templates:
  spike_artifacts:
    description: "Impulse-like spikes; watch smoothness/nonnegativity"
    affects: ["AIRS", "FGS1"]
    expected_rules: ["smoothness_l2", "nonnegativity"]
  band_mismatch:
    description: "Molecular band edges off alignment"
    affects: ["AIRS"]
    expected_rules: ["molecular_coherence", "photonic_alignment"]
""",
)

# Events

_add_file(
    "configs/symbolic/overrides/events/sample_events.yaml",
    "Event-level (per-planet / per-observation) symbolic overrides.",
    """# SpectraMind V50 — Event Overrides (sample)
events:
  planet_0001:
    profiles: ["default"]
    weights:
      molecular_coherence: 0.8
  planet_0002:
    profiles: ["kaggle_fast"]
    weights:
      smoothness_l2: 0.7
""",
)

# Instruments

_add_file(
    "configs/symbolic/overrides/instruments/airs.yaml",
    "AIRS-specific overrides.",
    """# SpectraMind V50 — Instrument Overrides (AIRS)
instrument: "AIRS"
weights:
  fft_asymmetry: 0.25
  photonic_alignment: 0.4
""",
)

_add_file(
    "configs/symbolic/overrides/instruments/fgs1.yaml",
    "FGS1-specific overrides.",
    """# SpectraMind V50 — Instrument Overrides (FGS1)
instrument: "FGS1"
weights:
  smoothness_l2: 1.2
""",
)

# Competition mode toggle

_add_file(
    "configs/symbolic/overrides/competition/neurips2025.yaml",
    "NeurIPS 2025 competition-mode symbolic toggles.",
    """# SpectraMind V50 — Competition Overrides (NeurIPS 2025)
competition:
  name: "neurips2025_ariel"
  submission:
    max_runtime_hours: 9
    device: "A100|L4"
    profiles: ["kaggle_fast"]
    weights:
      molecular_coherence: 0.4
""",
)

# ---- Surrounding config scaffolds & docs expected by the system ----

_add_dir("configs", "Global config root (Hydra).")
_add_dir("configs/calibration", "Calibration config root.")
_add_dir("configs/diagnostics", "Diagnostics config root.")
_add_dir("configs/model", "Model config root.")
_add_dir("configs/symbolic/molecules/_schemas", "Molecule schema definitions.")
_add_dir(".vscode", "VS Code settings (optional).")
_add_file(
    "configs/symbolic/README.md",
    "Human-focused doc for symbolic configuration.",
    """# Symbolic Configuration — SpectraMind V50

This folder contains **symbolic rules**, **overrides**, and **schemas** used by SpectraMind V50.

* `overrides/` holds profiles, weights, violations, event/instrument/competition packs.
* `molecules/` can define molecule-region metadata and schemas used by symbolic checks.
* `_schemas/` provides YAML/JSON schema templates for validation.

All files are **Hydra-safe** (no side effects on import), and can be layered via `defaults`.
""",
)

# Example diagnostic config

_add_file(
    "configs/diagnostics/base.yaml",
    "Diagnostics base config (Hydra-safe).",
    """# Diagnostics config (base)
diagnostics:
  umap:
    enabled: true
    dim: 2
  tsne:
    enabled: true
    perplexity: 30
  html_report:
    enabled: true
    outfile: "reports/diagnostic_report_v1.html"
""",
)

# Calibration base stub

_add_file(
    "configs/calibration/base.yaml",
    "Calibration base config (Hydra-safe).",
    """# Calibration base config
calibration:
  enable_corel: true
  temperature_scaling: true
  outputs_dir: "outputs/calibration"
""",
)

# VS Code helper (optional)

_add_file(
    ".vscode/settings.json",
    "VSCode defaults tuned for Hydra/YAML.",
    """{
  "files.associations": {
    "*.yaml": "yaml",
    "*.yml": "yaml"
  },
  "editor.rulers": [100],
  "editor.formatOnSave": true,
  "python.analysis.typeCheckingMode": "basic"
}
""",
)

# Root docs to nudge repository structure

_add_file(
    "ARCHITECTURE.md",
    "Architecture index (placeholder if missing).",
    """# SpectraMind V50 — Architecture

If you already have ARCHITECTURE.md in the repo, this file will NOT overwrite it.
This is a placeholder only, created by the auditor if missing.
""",
)

# -----------------------------

# Starter file content helpers

# -----------------------------


def _write_file(path: Path, content: str, logger: logging.Logger):
    if path.exists():
        logger.debug(f"Skip (exists): {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Created file: {path}")
    return True


# -----------------------------

# Auditor / Scaffolder

# -----------------------------


@dataclasses.dataclass
class Finding:
    path: str
    kind: str  # "missing_dir" | "missing_file" | "ok_dir" | "ok_file"
    description: str


def audit_repo(repo_root: Path, logger: logging.Logger) -> list[Finding]:
    findings: list[Finding] = []
    for rel, spec in sorted(MANIFEST.items(), key=lambda kv: kv[0]):
        abs_path = repo_root / rel
        if spec["type"] == "dir":
            if abs_path.is_dir():
                findings.append(Finding(rel, "ok_dir", spec["description"]))
            else:
                findings.append(Finding(rel, "missing_dir", spec["description"]))
        elif spec["type"] == "file":
            if abs_path.is_file():
                findings.append(Finding(rel, "ok_file", spec["description"]))
            else:
                findings.append(Finding(rel, "missing_file", spec["description"]))
    return findings


def apply_scaffold(repo_root: Path, findings: list[Finding], logger: logging.Logger):
    for f in findings:
        spec = MANIFEST.get(f.path, None)
        if not spec:
            continue
        abs_path = repo_root / f.path
        if f.kind == "missing_dir" and spec["type"] == "dir":
            abs_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {abs_path}")
        elif f.kind == "missing_file" and spec["type"] == "file":
            _write_file(abs_path, spec["template"], logger)


# -----------------------------

# Reporting

# -----------------------------


def write_reports(
    repo_root: Path, findings: list[Finding], logger: logging.Logger
) -> dict[str, Any]:
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "ts": _now_utc(),
        "summary": {
            "total": len(findings),
            "missing": sum(1 for f in findings if f.kind.startswith("missing")),
            "missing_dirs": sum(1 for f in findings if f.kind == "missing_dir"),
            "missing_files": sum(1 for f in findings if f.kind == "missing_file"),
        },
        "findings": [dataclasses.asdict(f) for f in findings],
        "git": _read_git(repo_root),
        "env": {
            "snapshot": _safe_env_snapshot(),
        },
        "python": _python_snapshot(),
    }

    # JSON
    out_json = reports_dir / "repo_audit_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote JSON report: {out_json}")

    return report


# -----------------------------

# Git ops

# -----------------------------


def git_commit_and_push(repo_root: Path, logger: logging.Logger, branch: str | None = None):
    def run(cmd: list[str], check=True):
        logger.debug(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(repo_root), check=check)

    if branch is None:
        branch = f"chore/repo-audit-{int(time.time())}"

    # Create branch (if not already on it)
    try:
        run(["git", "checkout", "-b", branch])
    except subprocess.CalledProcessError:
        # If exists, checkout
        run(["git", "checkout", branch])

    run(["git", "add", "-A"])
    run(
        [
            "git",
            "commit",
            "-m",
            "chore(repo): add missing symbolic overrides & configs (auto-scaffold)",
        ]
    )
    # Push to origin (assumes origin exists and auth is set)
    run(["git", "push", "-u", "origin", branch])
    return branch


# -----------------------------

# MLflow/W&B (optional stubs)

# -----------------------------


def _maybe_mlflow_start(logger: logging.Logger):
    if os.environ.get("SMV50_MLFLOW_ENABLE", "") != "1":
        return None
    try:
        import mlflow

        mlflow.set_experiment("SpectraMindV50-RepoAudit")
        mlflow.start_run(run_name=f"audit-{int(time.time())}")
        logger.info("MLflow run started.")
        return mlflow
    except Exception as e:
        logger.warning(f"MLflow not available: {e}")
        return None


def _maybe_wandb_start(logger: logging.Logger):
    if os.environ.get("SMV50_WANDB_ENABLE", "") != "1":
        return None
    try:
        import wandb

        wandb.init(project="SpectraMindV50-RepoAudit", name=f"audit-{int(time.time())}")
        logger.info("W&B run started.")
        return wandb
    except Exception as e:
        logger.warning(f"W&B not available: {e}")
        return None


# -----------------------------

# Main

# -----------------------------


def main():  # noqa: PLR0912, PLR0915
    ap = argparse.ArgumentParser(description="SpectraMind V50 Repo Auditor & Scaffolder")
    ap.add_argument("--repo", type=str, default=".", help="Path to repo root.")
    ap.add_argument("--apply", action="store_true", help="Create missing files/dirs per manifest.")
    ap.add_argument("--git", action="store_true", help="Commit & push changes on a new branch.")
    ap.add_argument("--branch", type=str, default=None, help="Branch name to use (with --git).")
    ap.add_argument("--verbose", action="store_true", help="Verbose console logging.")
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()
    repo_root.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(repo_root, verbose=args.verbose)
    emitter = JSONLEmitter(repo_root / "reports" / "repo_audit_events.jsonl")

    logger.info("SpectraMind V50 — Repo Auditor & Scaffolder")
    logger.info(f"Repo root: {repo_root}")

    mlflow = _maybe_mlflow_start(logger)
    wandb = _maybe_wandb_start(logger)

    try:
        findings = audit_repo(repo_root, logger)

        # Emit to JSONL
        for f in findings:
            emitter.emit({"event": "finding", **dataclasses.asdict(f)})

        # Human-friendly console summary
        missing = [f for f in findings if f.kind.startswith("missing")]
        ok = [f for f in findings if f.kind.startswith("ok")]
        if missing:
            logger.info("\n=== MISSING ITEMS ===")
            for f in missing:
                logger.info(
                    f"- {f.kind.replace('_', ' ').upper():14s} :: {f.path} — {f.description}"
                )
        else:
            logger.info("No missing items per current manifest.")

        logger.info("\n=== PRESENT ITEMS ===")
        for f in ok[:MAX_PRESENT_ITEMS]:
            logger.info(f"- {f.kind.replace('_', ' ').upper():14s} :: {f.path}")
        if len(ok) > MAX_PRESENT_ITEMS:
            logger.info(f"... and {len(ok) - MAX_PRESENT_ITEMS} more present items")

        report = write_reports(repo_root, findings, logger)

        if args.apply:
            logger.info("\nApplying scaffolding for missing items...")
            apply_scaffold(repo_root, findings, logger)
            emitter.emit({"event": "apply_done", "ts": _now_utc()})
            logger.info("Scaffolding complete.")

            # Re-run audit to show final state
            findings2 = audit_repo(repo_root, logger)
            write_reports(repo_root, findings2, logger)

        # Git operations (after apply, if chosen)
        if args.git:
            try:
                branch = git_commit_and_push(repo_root, logger, branch=args.branch)
                emitter.emit({"event": "git_push", "branch": branch, "ts": _now_utc()})
                logger.info(f"Changes pushed on branch: {branch}")
            except Exception as e:
                logger.error(f"Git push failed: {e}")
                emitter.emit({"event": "git_push_failed", "error": str(e), "ts": _now_utc()})

        # Log end
        emitter.emit({"event": "done", "ts": _now_utc()})

        # Log to MLflow/W&B minimal metadata
        if mlflow:
            try:
                import mlflow

                mlflow.log_dict(report, "repo_audit_report.json")
                mlflow.log_metric("missing_total", report["summary"]["missing"])
                mlflow.log_metric("missing_dirs", report["summary"]["missing_dirs"])
                mlflow.log_metric("missing_files", report["summary"]["missing_files"])
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        if wandb:
            try:
                import wandb

                wandb.log(
                    {
                        "missing_total": report["summary"]["missing"],
                        "missing_dirs": report["summary"]["missing_dirs"],
                        "missing_files": report["summary"]["missing_files"],
                    }
                )
                wandb.finish()
            except Exception as e:
                logger.warning(f"W&B logging failed: {e}")

        logger.info("\nDone.")
    finally:
        emitter.close()


if __name__ == "__main__":
    main()
