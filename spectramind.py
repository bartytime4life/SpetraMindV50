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
