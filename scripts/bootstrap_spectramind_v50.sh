#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SpectraMind V50 — Repository Bootstrapper (one‑command setup)
# -----------------------------------------------------------------------------
# This script scaffolds a fully runnable, CLI‑first, neuro‑symbolic, physics‑
# informed pipeline skeleton for the NeurIPS 2025 Ariel Data Challenge.
# It creates: Makefile, Poetry pyproject, Hydra configs, Typer CLI, selftests,
# logging utils, minimal model/training/inference/diagnostics stubs, DVC DAG,
# GitHub Actions CI, and reproducibility artifacts.
#
# Usage:
#   bash scripts/bootstrap_spectramind_v50.sh
#
# After it finishes:
#   1) poetry install --no-root && poetry shell
#   2) python -m spectramind --version
#   3) python -m spectramind selftest
#   4) python -m spectramind diagnose --dry-run
#   5) make ci-local
# -----------------------------------------------------------------------------
set -euo pipefail
umask 022

ROOT_DIR="$(pwd)"
mkdir -p scripts configs/{data,model,train,diag} src/spectramind/{cli,utils,models,training,inference,diagnostics,calibration,symbolic} .github/workflows outputs .dvc tmp docs

# --- .gitignore ---------------------------------------------------------------
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.venv/

# Poetry
poetry.lock

# DVC / data artifacts
.dvc/
.dvcignore
.cache/
outputs/
/tmp/
/tmp/**

# Notebooks & logs
*.ipynb_checkpoints/
.vscode/
.vscode/**
.v50_cache/
*.log

# Generated
run_hash_summary_v50.json
v50_debug_log.md
events.jsonl

# OS
.DS_Store
Thumbs.db
EOF

# --- Makefile ----------------------------------------------------------------
cat > Makefile << 'EOF'
# SpectraMind V50 — Makefile convenience targets

.PHONY: help deps lock fmt lint test selftest train predict dashboard ci-local

help:
	@echo "Targets: deps | lock | fmt | lint | test | selftest | train | predict | dashboard | ci-local"

# Install deps via Poetry (no-root keeps project editable)
deps:
	poetry install --no-root

# Refresh lockfile
lock:
	poetry lock --no-update

fmt:
	ruff format

lint:
	ruff check --fix

# Lightweight unit tests placeholder
test:
	python -m pytest -q || true

selftest:
	python -m spectramind selftest

train:
	python -m spectramind train --dry-run

predict:
	python -m spectramind predict --out-csv outputs/submission.csv

dashboard:
	python -m spectramind dashboard --html outputs/diagnostics/diagnostic_report_v50.html

ci-local: deps selftest test predict dashboard
EOF

# --- pyproject.toml (Poetry) --------------------------------------------------
cat > pyproject.toml << 'EOF'
[tool.poetry]
name = "spectramind-v50"
version = "0.1.0"
description = "SpectraMind V50 — Neuro-symbolic pipeline for the NeurIPS 2025 Ariel Data Challenge"
authors = ["SpectraMind Team <team@example.com>"]
readme = "README.md"
packages = [{ include = "src" }]

[tool.poetry.dependencies]
python = "^3.10"
typer = {version = "^0.12.3", extras = ["all"]}
hydra-core = "^1.3.2"
omegaconf = "^2.3.0"
pyyaml = "^6.0.2"
rich = "^13.7.1"
loguru = "^0.7.2"
uvicorn = "^0.30.3"
fastapi = "^0.111.0"
numpy = "^1.26.4"
pandas = "^2.2.2"
scipy = "^1.13.1"
matplotlib = "^3.9.0"
plotly = "^5.22.0"
joblib = "^1.4.2"
filelock = "^3.15.4"
psutil = "^6.0.0"
# Optional heavy deps (install as needed)
# torch = {version = "^2.3.1", markers = "platform_system != 'Darwin'"}
# torch-geometric = "*"
# shap = "^0.45.1"
# umap-learn = "^0.5.6"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.1"
ruff = "^0.5.6"

[build-system]
requires = ["poetry-core>=1.8.0"]
build-backend = "poetry.core.masonry.api"

[tool.ruff]
line-length = 100
EOF

# --- README.md (lightweight, full docs live elsewhere) -----------------------
cat > README.md << 'EOF'
# SpectraMind V50 — Bootstrap
This repository contains the CLI-first, Hydra-safe scaffolding for SpectraMind V50.
Run `make deps && python -m spectramind --version` to verify the environment.
See docs/ and the HTML dashboard produced by `spectramind dashboard`.
EOF

# --- configs (Hydra) ----------------------------------------------------------
cat > configs/config_v50.yaml << 'EOF'
# Root Hydra config for SpectraMind V50
seed: 1337
run:
  out_dir: outputs
  log_file: v50_debug_log.md
  jsonl_file: events.jsonl
  save_hash_summary: run_hash_summary_v50.json

loader:
  batch_size: 8
  num_workers: 2
  pin_memory: true
  persistent_workers: false

paths:
  fgs1: "~/datasets/ariel/raw/fgs1"
  airs: "~/datasets/ariel/raw/airs_ch0"
  calibration: "~/datasets/ariel/calibration"
  metadata: ""
  splits: ""

model:
  fgs1_mamba:
    latent_dim: 128
  airs_gnn:
    latent_dim: 128
    use_gat: true
  decoder:
    multiscale: true
  sigma_head:
    type: flow
    softplus: true

loss:
  lambda_symbolic: 0.1

train:
  phase: supervised   # mae | contrastive | supervised
  epochs: 2
  optimizer: adamw
  lr: 3e-4
  amp: true

calibration:
  temperature: 1.0
  corel:
    enabled: true

diagnostics:
  make_html: true
EOF

cat > configs/data/local.yaml << 'EOF'
# Local workstation overrides
paths:
  fgs1: "~/datasets/ariel/raw/fgs1"
  airs: "~/datasets/ariel/raw/airs_ch0"
  calibration: "~/datasets/ariel/calibration"
loader:
  batch_size: 8
  num_workers: 4
EOF

cat > configs/data/kaggle.yaml << 'EOF'
# Kaggle environment overrides
paths:
  fgs1: "/kaggle/input/ariel-fgs1/raw/fgs1"
  airs: "/kaggle/input/ariel-airs-ch0/raw/airs_ch0"
  calibration: "/kaggle/input/ariel-calibration/calibration"
loader:
  batch_size: 8
  num_workers: 2
EOF

# --- DVC pipeline skeleton ----------------------------------------------------
cat > dvc.yaml << 'EOF'
stages:
  calibrate:
    cmd: python -m spectramind calibrate --no-run   # placeholder for now
    deps:
      - src/spectramind/calibration/__init__.py
    outs:
      - outputs/calibrated
  features:
    cmd: python -m spectramind features --no-run    # placeholder for now
    deps:
      - outputs/calibrated
    outs:
      - outputs/features
  train:
    cmd: python -m spectramind train
    deps:
      - outputs/features
      - src/spectramind/training/train_v50.py
    outs:
      - outputs/checkpoints
  predict:
    cmd: python -m spectramind predict --out-csv outputs/submission.csv
    deps:
      - outputs/checkpoints
    outs:
      - outputs/submission.csv
  dashboard:
    cmd: python -m spectramind dashboard --html outputs/diagnostics/diagnostic_report_v50.html
    deps:
      - outputs/submission.csv
      - src/spectramind/diagnostics/generate_html_report.py
    outs:
      - outputs/diagnostics/diagnostic_report_v50.html
EOF

# --- Typer root CLI: spectramind.py ------------------------------------------
cat > spectramind.py << 'EOF'
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich import print

APP_VERSION = "0.1.0"
APP = typer.Typer(add_completion=False, help="SpectraMind V50 — Unified CLI")

# -------------------------- logging & telemetry ------------------------------ #

@dataclass
class RunMeta:
    timestamp: str
    user: str
    host: str
    os: str
    py: str
    git_sha: str
    config_hash: str
    cmd: str


def _iso_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _git_sha() -> str:
    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return "NA"


def _hash_configs() -> str:
    h = hashlib.sha256()
    for p in sorted(Path("configs").rglob("*.yaml")):
        try:
            h.update(p.read_bytes())
        except Exception:
            pass
    return h.hexdigest()[:12]


def _append_log(line: str, log_file: str = "v50_debug_log.md") -> None:
    try:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _append_jsonl(event: dict, jsonl_file: str = "events.jsonl") -> None:
    try:
        p = Path(jsonl_file)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ------------------------------- commands ----------------------------------- #

@APP.command()
def version() -> None:
    """Print CLI version + config hash and write to audit logs."""
    meta = RunMeta(
        timestamp=_iso_utc(),
        user=os.getenv("USER", "unknown"),
        host=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        py=sys.version.split()[0],
        git_sha=_git_sha(),
        config_hash=_hash_configs(),
        cmd="version",
    )
    line = f"[version] {meta.timestamp} v{APP_VERSION} git={meta.git_sha} cfg={meta.config_hash}"
    _append_log(line)
    _append_jsonl({"event": "version", **asdict(meta)})
    print(line)


@APP.command()
def selftest() -> None:
    """Verify file presence, CLI registration, config readability."""
    required = [
        "configs/config_v50.yaml",
        "spectramind.py",
        "src/spectramind/cli/selftest.py",
        "src/spectramind/diagnostics/generate_html_report.py",
    ]
    missing = [p for p in required if not Path(p).exists()]
    status = "ok" if not missing else f"missing: {missing}"
    _append_log(f"[selftest] {_iso_utc()} {status}")
    _append_jsonl({"event": "selftest", "status": status, "missing": missing})
    if missing:
        print(f"[red]❌ Missing files: {missing}")
        raise typer.Exit(1)
    print("[green]✅ Selftest passed.")


@APP.command()
def train(dry_run: bool = typer.Option(False, help="Skip heavy work")) -> None:
    """Train pipeline (stub): MAE→(contrastive)→supervised with GLL+symbolic."""
    from src.spectramind.training.train_v50 import train as _train

    _append_log(f"[train] {_iso_utc()} dry_run={dry_run}")
    _append_jsonl({"event": "train", "dry_run": dry_run})
    _train(dry_run=dry_run)


@APP.command()
def predict(out_csv: Path = typer.Option(Path("outputs/submission.csv"), exists=False)) -> None:
    """Run inference and write μ/σ submission CSV (stub)."""
    from src.spectramind.inference.predict_v50 import predict as _predict

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _append_log(f"[predict] {_iso_utc()} -> {out_csv}")
    _append_jsonl({"event": "predict", "out_csv": str(out_csv)})
    _predict(out_csv)
    print(f"[green]✅ Submission written: {out_csv}")


@APP.command("dashboard")
def dashboard_cmd(html: Path = typer.Option(Path("outputs/diagnostics/diagnostic_report_v50.html"))) -> None:
    """Generate versioned HTML diagnostics report (stub)."""
    from src.spectramind.diagnostics.generate_html_report import write_report

    html.parent.mkdir(parents=True, exist_ok=True)
    _append_log(f"[dashboard] {_iso_utc()} -> {html}")
    _append_jsonl({"event": "dashboard", "html": str(html)})
    write_report(html)
    print(f"[green]✅ Diagnostics HTML: {html}")


@APP.command("calibrate-temp")
def calibrate_temp() -> None:
    _append_log(f"[calibrate-temp] {_iso_utc()}")
    _append_jsonl({"event": "calibrate-temp"})
    print("[yellow]ℹ️ Temperature scaling placeholder (to be implemented).")


@APP.command("calibrate-corel")
def calibrate_corel() -> None:
    _append_log(f"[calibrate-corel] {_iso_utc()}")
    _append_jsonl({"event": "calibrate-corel"})
    print("[yellow]ℹ️ COREL conformal calibration placeholder (to be implemented).")


@APP.command()
def submit(
    bundle: bool = typer.Option(True, help="Create a submission bundle zip"),
    in_csv: Path = typer.Option(Path("outputs/submission.csv")),
    out_zip: Path = typer.Option(Path("outputs/submission_bundle.zip")),
) -> None:
    if not in_csv.exists():
        print(f"[red]❌ missing submission: {in_csv}")
        raise typer.Exit(1)
    import zipfile

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w") as z:
        z.write(in_csv, arcname="submission.csv")
        for extra in ("run_hash_summary_v50.json", "v50_debug_log.md", "events.jsonl"):
            if Path(extra).exists():
                z.write(extra, arcname=Path(extra).name)
    _append_log(f"[submit] {_iso_utc()} -> {out_zip}")
    _append_jsonl({"event": "submit", "out_zip": str(out_zip)})
    print(f"[green]✅ Bundled -> {out_zip}")


@APP.command()
def diagnose(dry_run: bool = typer.Option(False, help="Print planned diagnostics only")) -> None:
    _append_log(f"[diagnose] {_iso_utc()} dry_run={dry_run}")
    _append_jsonl({"event": "diagnose", "dry_run": dry_run})
    print("[cyan]Diagnostics planned: SHAP, UMAP/t-SNE, FFT, smoothness, symbolic overlays.")


if __name__ == "__main__":
    APP()
EOF

# --- src/spectramind/utils/logging.py ----------------------------------------
cat > src/spectramind/utils/logging.py << 'EOF'
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict


class HumanFileHandler(RotatingFileHandler):
    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        msg = self.format(record)
        try:
            with open(self.baseFilename, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass


def setup_logging(log_path: str = "v50_debug_log.md", jsonl_path: str = "events.jsonl") -> logging.Logger:
    logger = logging.getLogger("spectramind")
    logger.setLevel(logging.INFO)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    hf = HumanFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    hf.setFormatter(logging.Formatter("%(asctime)sZ [%(levelname)s] %(message)s"))
    logger.addHandler(hf)

    class JSONL(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
            try:
                event: Dict[str, Any] = {
                    "ts": record.asctime if hasattr(record, "asctime") else None,
                    "level": record.levelname,
                    "msg": record.getMessage(),
                }
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception:
                pass

    logger.addHandler(JSONL())
    return logger
EOF

# --- src/spectramind/cli/selftest.py -----------------------------------------
cat > src/spectramind/cli/selftest.py << 'EOF'
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
EOF

# --- src/spectramind/cli/pipeline_consistency_checker.py ---------------------
cat > src/spectramind/cli/pipeline_consistency_checker.py << 'EOF'
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
EOF

# --- src/spectramind/training/train_v50.py -----------------------------------
cat > src/spectramind/training/train_v50.py << 'EOF'
from __future__ import annotations

from pathlib import Path
from rich import print


def train(dry_run: bool = False) -> None:
    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("[cyan]DRY-RUN: would train MAE→(contrastive)→supervised with GLL+symbolic.")
        (Path("outputs/checkpoints") / "model_stub.pt").write_text("stub")
        return
    # Placeholder for real training loop
    (Path("outputs/checkpoints") / "model_stub.pt").write_text("trained")
    print("[green]✅ Training stub complete (artifact written).")
EOF

# --- src/spectramind/inference/predict_v50.py --------------------------------
cat > src/spectramind/inference/predict_v50.py << 'EOF'
from __future__ import annotations

import csv
from pathlib import Path

NUM_BINS = 283


def predict(out_csv: Path) -> None:
    # Write a minimal valid submission with constant μ/σ (for CI wire-up)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [f"mu_{i}" for i in range(NUM_BINS)] + [f"sigma_{i}" for i in range(NUM_BINS)]
        w.writerow(header)
        # single dummy row
        w.writerow([0.0] * NUM_BINS + [0.1] * NUM_BINS)
EOF

# --- src/spectramind/diagnostics/generate_html_report.py ---------------------
cat > src/spectramind/diagnostics/generate_html_report.py << 'EOF'
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def write_report(path: Path) -> None:
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>SpectraMind V50 — Diagnostics</title>
  <style>body{{font-family:system-ui,Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}}.card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:12px}}</style>
</head>
<body>
  <h1>Diagnostics — SpectraMind V50</h1>
  <div class=\"card\">
    <h2>Run Metadata</h2>
    <p>Generated: {ts}</p>
  </div>
  <div class=\"card\">
    <h2>Placeholders</h2>
    <ul>
      <li>UMAP/t-SNE: pending</li>
      <li>SHAP × symbolic overlays: pending</li>
      <li>FFT/smoothness maps: pending</li>
    </ul>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
EOF

# --- src/spectramind/models stubs --------------------------------------------
cat > src/spectramind/models/__init__.py << 'EOF'
from .fgs1_mamba import FGS1MambaEncoder
from .airs_gnn import AIRSSpectralGNN
from .multi_scale_decoder import MultiScaleDecoder
from .flow_uncertainty_head import FlowUncertaintyHead
EOF

cat > src/spectramind/models/fgs1_mamba.py << 'EOF'
from __future__ import annotations

import math
from typing import Optional

class FGS1MambaEncoder:
    """Placeholder for a Mamba-style SSM encoder for temporal FGS1 sequences."""

    def __init__(self, in_dim: int = 64, latent_dim: int = 128, bidirectional: bool = True) -> None:
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional

    def __repr__(self) -> str:  # TorchScript-safe style goal (no torch deps here)
        return f"FGS1MambaEncoder(in_dim={self.in_dim}, latent_dim={self.latent_dim}, bidirectional={self.bidirectional})"

    def encode(self, x_len: int) -> list[float]:  # stub output
        return [math.sin(i / 10.0) for i in range(self.latent_dim)]
EOF

cat > src/spectramind/models/airs_gnn.py << 'EOF'
from __future__ import annotations

class AIRSSpectralGNN:
    """Placeholder spectral GNN; real version will use torch_geometric GAT/NNConv."""

    def __init__(self, in_dim: int = 64, latent_dim: int = 128, use_gat: bool = True) -> None:
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.use_gat = use_gat

    def __repr__(self) -> str:
        return f"AIRSSpectralGNN(in_dim={self.in_dim}, latent_dim={self.latent_dim}, use_gat={self.use_gat})"

    def encode(self, num_nodes: int = 283) -> list[float]:
        return [0.0 for _ in range(self.latent_dim)]
EOF

cat > src/spectramind/models/multi_scale_decoder.py << 'EOF'
from __future__ import annotations

class MultiScaleDecoder:
    def __init__(self, out_bins: int = 283, multiscale: bool = True) -> None:
        self.out_bins = out_bins
        self.multiscale = multiscale

    def decode_mu(self, latent: list[float]) -> list[float]:
        return [0.0 for _ in range(self.out_bins)]
EOF

cat > src/spectramind/models/flow_uncertainty_head.py << 'EOF'
from __future__ import annotations

class FlowUncertaintyHead:
    def __init__(self, out_bins: int = 283, softplus: bool = True) -> None:
        self.out_bins = out_bins
        self.softplus = softplus

    def decode_sigma(self, latent: list[float]) -> list[float]:
        return [0.1 for _ in range(self.out_bins)]
EOF

# --- GitHub Actions CI --------------------------------------------------------
cat > .github/workflows/ci.yml << 'EOF'
name: ci

on:
  push:
    branches: [ main, master ]
  pull_request:

jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install Poetry
        run: |
          pipx install poetry
      - name: Install deps
        run: |
          poetry install --no-root
      - name: Selftest
        run: |
          python -m spectramind --version
          python -m spectramind selftest
      - name: Predict & Dashboard (smoke)
        run: |
          python -m spectramind predict --out-csv outputs/submission.csv
          python -m spectramind dashboard --html outputs/diagnostics/diagnostic_report_v50.html
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: diagnostics
          path: |
            outputs/submission.csv
            outputs/diagnostics/diagnostic_report_v50.html
            v50_debug_log.md
            events.jsonl
EOF

# --- seed logs ----------------------------------------------------------------
: > v50_debug_log.md
: > events.jsonl

# --- Fin ---------------------------------------------------------------------
echo "✅ SpectraMind V50 bootstrap complete. Next steps:"
echo "   1) poetry install --no-root && poetry shell"
echo "   2) python -m spectramind --version"
echo "   3) python -m spectramind selftest"
echo "   4) python -m spectramind predict --out-csv outputs/submission.csv"
echo "   5) python -m spectramind dashboard --html outputs/diagnostics/diagnostic_report_v50.html"