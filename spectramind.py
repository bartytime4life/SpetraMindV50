#!/usr/bin/env python3
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

# Repo-relative paths
ROOT = Path(__file__).resolve().parent
REPO = ROOT
OUT = REPO / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DEBUG_LOG = REPO / "v50_debug_log.md"
EVENTS_PATH = REPO / "events.jsonl"


# ---------------------------
# Event logger (JSONL)
# ---------------------------
class EventLogger:
    """
    Structured JSONL event logger for SpectraMind V50.
    Writes one JSON object per event to events.jsonl (ensure_ascii=True).
    Also exposes a context manager to time steps and emit start/end markers.
    """

    def __init__(self, jsonl_path: str | Path = "events.jsonl", run_id: Optional[str] = None):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self._gen_run_id()

    @staticmethod
    def _gen_run_id() -> str:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        suf = f"{int.from_bytes(os.urandom(3), 'big'):06x}"
        return f"{ts}-{suf}"

    def emit(
        self,
        *,
        phase: str,
        step: str,
        component: str,
        params: Dict[str, Any] | None = None,
        metrics: Dict[str, Any] | None = None,
        duration_ms: int | None = None,
        git_commit: str = "",
        config_hash: str = "",
        cli_cmd: str = "",
    ) -> None:
        record = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "phase": phase,
            "step": step,
            "component": component,
            "params": params or {},
            "metrics": metrics or {},
            "duration_ms": int(duration_ms or 0),
            "git_commit": git_commit,
            "config_hash": config_hash,
            "cli_cmd": cli_cmd,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def timer(
        self,
        *,
        phase: str,
        step: str,
        component: str,
        git_commit: str,
        config_hash: str,
        cli_cmd: str,
    ):
        return _EventTimer(self, phase, step, component, git_commit, config_hash, cli_cmd)


class _EventTimer:
    def __init__(
        self,
        logger: EventLogger,
        phase: str,
        step: str,
        component: str,
        git_commit: str,
        config_hash: str,
        cli_cmd: str,
    ):
        self.logger = logger
        self.phase = phase
        self.step = step
        self.component = component
        self.git_commit = git_commit
        self.config_hash = config_hash
        self.cli_cmd = cli_cmd
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        self.logger.emit(
            phase=self.phase,
            step=self.step + ".start",
            component=self.component,
            git_commit=self.git_commit,
            config_hash=self.config_hash,
            cli_cmd=self.cli_cmd,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        dur_ms = int((time.time() - self.t0) * 1000.0)
        step = self.step + (".error" if exc_type is not None else ".end")
        metrics = {"exception": (exc_type.__name__ if exc_type else "")}
        self.logger.emit(
            phase=self.phase,
            step=step,
            component=self.component,
            metrics=metrics,
            duration_ms=dur_ms,
            git_commit=self.git_commit,
            config_hash=self.config_hash,
            cli_cmd=self.cli_cmd,
        )
        return False  # do not swallow exceptions


# ---------------------------
# Helpers
# ---------------------------
def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO)
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def _load_cfg(overrides: Optional[List[str]] = None):
    cfg_path = REPO / "configs" / "config_v50.yaml"
    if not cfg_path.exists():
        # return minimal config so --version and selftest can show something
        base = OmegaConf.create(
            {
                "project": {"name": "spectramind-v50", "seed": 42},
                "runtime": {"device": "auto", "num_workers": 4, "fp16": True},
                "logging": {
                    "level": "INFO",
                    "rich": True,
                    "audit_path": str(DEBUG_LOG),
                    "jsonl_path": str(EVENTS_PATH),
                },
            }
        )
        if overrides:
            for ov in overrides:
                if "=" in ov:
                    k, v = ov.split("=", 1)
                    OmegaConf.update(base, k, v, merge=True)
        return base

    base = OmegaConf.load(cfg_path)
    if overrides:
        for ov in overrides:
            if "=" in ov:
                k, v = ov.split("=", 1)
                OmegaConf.update(base, k, v, merge=True)
    return base


def _config_hash(cfg) -> str:
    try:
        s = OmegaConf.to_yaml(cfg)
    except Exception:
        s = str(cfg)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _append_audit(line: str) -> None:
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def _header(cfg) -> None:
    table = Table(title="SpectraMind V50 - Run Header")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Time (UTC)", datetime.datetime.utcnow().isoformat())
    table.add_row("Host", platform.node())
    table.add_row("OS", f"{platform.system()} {platform.release()}")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Git", _git_sha())
    table.add_row("Config Hash", _config_hash(cfg))
    console.print(table)


def _stamp(cmd: str, cfg, logger: EventLogger) -> None:
    _append_audit(
        f"- {datetime.datetime.utcnow().isoformat()}  `{cmd}`  git={_git_sha()} cfg={_config_hash(cfg)}"
    )
    logger.emit(
        phase="cli",
        step="invoke",
        component="spectramind.cli",
        params={"cmd": cmd},
        git_commit=_git_sha(),
        config_hash=_config_hash(cfg),
        cli_cmd=cmd,
    )


# ---------------------------
# CLI commands
# ---------------------------
@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show version and provenance."),
):
    if version:
        cfg = _load_cfg([])
        logger = EventLogger(EVENTS_PATH)
        _header(cfg)
        _stamp("spectramind --version", cfg, logger)
        console.print(Panel.fit("SpectraMind V50 CLI - version stub", style="bold green"))
        raise typer.Exit()


@app.command()
def selftest(overrides: List[str] = typer.Argument(None, help="Hydra-style overrides: key=value ...")):
    """
    Verify key files, CLI registration, and ability to write logs.
    """
    cfg = _load_cfg(overrides or [])
    logger = EventLogger(EVENTS_PATH)
    _header(cfg)
    _stamp("spectramind selftest", cfg, logger)

    cmd = "spectramind selftest"
    with logger.timer(
        phase="selftest",
        step="run",
        component="spectramind.selftest",
        git_commit=_git_sha(),
        config_hash=_config_hash(cfg),
        cli_cmd=cmd,
    ):
        required = [
            "configs/config_v50.yaml",
            "schemas/submission.schema.json",
        ]
        missing = [p for p in required if not (REPO / p).exists()]
        logger.emit(
            phase="selftest",
            step="check.required_files",
            component="spectramind.selftest",
            params={"required": required},
            metrics={"missing": len(missing)},
            git_commit=_git_sha(),
            config_hash=_config_hash(cfg),
            cli_cmd=cmd,
        )
        if missing:
            console.print(f"[red]Missing required files: {missing}[/red]")
            raise typer.Exit(1)
        console.print("[green]Selftest OK[/green]")


@app.command()
def train(overrides: List[str] = typer.Argument(None)):
    """
    Run training (MAE -> supervised GLL with symbolic constraints).
    This stub just demonstrates logging and timing.
    """
    cfg = _load_cfg(overrides or [])
    logger = EventLogger(EVENTS_PATH)
    _header(cfg)
    _stamp("spectramind train", cfg, logger)

    cmd = "spectramind train"
    with logger.timer(
        phase="training",
        step="run",
        component="spectramind.training",
        git_commit=_git_sha(),
        config_hash=_config_hash(cfg),
        cli_cmd=cmd,
    ):
        console.print("Training stub - wire to src/spectramind/training/train_v50.py")
        time.sleep(0.05)
        logger.emit(
            phase="training",
            step="epoch_end",
            component="spectramind.training",
            params={"epoch": 0, "lr": 1e-3},
            metrics={"loss_gll": 0.0, "loss_symbolic": 0.0},
            duration_ms=50,
            git_commit=_git_sha(),
            config_hash=_config_hash(cfg),
            cli_cmd=cmd,
        )


@app.command("calibrate-temp")
def calibrate_temp(overrides: List[str] = typer.Argument(None)):
    """
    Temperature scaling for sigma calibration (stub).
    """
    cfg = _load_cfg(overrides or [])
    logger = EventLogger(EVENTS_PATH)
    _header(cfg)
    _stamp("spectramind calibrate-temp", cfg, logger)

    cmd = "spectramind calibrate-temp"
    with logger.timer(
        phase="calibration",
        step="temperature_scaling",
        component="spectramind.calibration.temperature",
        git_commit=_git_sha(),
        config_hash=_config_hash(cfg),
        cli_cmd=cmd,
    ):
        console.print("Temp scaling stub - wire to src/spectramind/calibration/temperature.py")
        time.sleep(0.05)


@app.command("calibrate-corel")
def calibrate_corel(overrides: List[str] = typer.Argument(None)):
    """
    COREL conformal calibration for spectral coverage control (stub).
    """
    cfg = _load_cfg(overrides or [])
    logger = EventLogger(EVENTS_PATH)
    _header(cfg)
    _stamp("spectramind calibrate-corel", cfg, logger)

    cmd = "spectramind calibrate-corel"
    with logger.timer(
        phase="calibration",
        step="corel",
        component="spectramind.calibration.corel",
        git_commit=_git_sha(),
        config_hash=_config_hash(cfg),
        cli_cmd=cmd,
    ):
        console.print("COREL stub - wire to src/spectramind/calibration/corel.py")
        time.sleep(0.05)


@app.command()
def diagnose(overrides: List[str] = typer.Argument(None)):
    """
    Generate diagnostics bundle (smoothness, SHAP overlays, coverage) (stub).
    """
    cfg = _load_cfg(overrides or [])
    logger = EventLogger(EVENTS_PATH)
    _header(cfg)
    _stamp("spectramind diagnose", cfg, logger)

    cmd = "spectramind diagnose"
    with logger.timer(
        phase="diagnostics",
        step="bundle",
        component="spectramind.diagnostics",
        git_commit=_git_sha(),
        config_hash=_config_hash(cfg),
        cli_cmd=cmd,
    ):
        console.print("Diagnostics stub - wire to src/spectramind/diagnostics/")
        time.sleep(0.05)


@app.command()
def predict(
    out_csv: Path = typer.Option(
        "outputs/submission.csv", "--out-csv", help="Where to write submission.csv"
    ),
    overrides: List[str] = typer.Argument(None),
):
    """
    Predict mu/sigma and emit submission CSV header (stub).
    """
    cfg = _load_cfg(overrides or [])
    logger = EventLogger(EVENTS_PATH)
    cmd = f"spectramind predict --out-csv {out_csv}"
    _header(cfg)
    _stamp(cmd, cfg, logger)

    with logger.timer(
        phase="inference",
        step="predict",
        component="spectramind.predict",
        git_commit=_git_sha(),
        config_hash=_config_hash(cfg),
        cli_cmd=cmd,
    ):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            header = (
                "planet_id,"
                + ",".join([f"mu_{i}" for i in range(283)])
                + ","
                + ",".join([f"sigma_{i}" for i in range(283)])
                + "\n"
            )
            f.write(header)
        console.print(f"Wrote submission stub: {out_csv}")
        logger.emit(
            phase="inference",
            step="write_csv",
            component="spectramind.predict",
            params={"path": str(out_csv)},
            metrics={"rows": 0},
            git_commit=_git_sha(),
            config_hash=_config_hash(cfg),
            cli_cmd=cmd,
        )


@app.command()
def submit(
    what: str = typer.Argument("bundle", help="`bundle` to zip submission + provenance."),
    overrides: List[str] = typer.Argument(None),
):
    """
    Create a submission bundle (submission + provenance).
    """
    cfg = _load_cfg(overrides or [])
    logger = EventLogger(EVENTS_PATH)
    _header(cfg)
    _stamp("spectramind submit bundle", cfg, logger)

    cmd = "spectramind submit bundle"
    with logger.timer(
        phase="packaging",
        step="bundle",
        component="spectramind.submit",
        git_commit=_git_sha(),
        config_hash=_config_hash(cfg),
        cli_cmd=cmd,
    ):
        zip_path = OUT / "submission_bundle.zip"
        import zipfile

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in ["outputs/submission.csv", "v50_debug_log.md", "events.jsonl"]:
                path = REPO / p
                if path.exists():
                    z.write(path, arcname=path.name)
        console.print(f"Bundle: {zip_path}")
        logger.emit(
            phase="packaging",
            step="zip",
            component="spectramind.submit",
            params={"zip_path": str(zip_path)},
            metrics={"files": 3},
            git_commit=_git_sha(),
            config_hash=_config_hash(cfg),
            cli_cmd=cmd,
        )


@app.command()
def ablate(overrides: List[str] = typer.Argument(None)):
    """
    Run ablations (disable components via overrides).
    Example:
      python -m spectramind ablate model.fusion.type=concat
    """
    cfg = _load_cfg(overrides or [])
    logger = EventLogger(EVENTS_PATH)
    cmd = "spectramind ablate " + " ".join(overrides or []) if overrides else "spectramind ablate"
    _header(cfg)
    _stamp(cmd, cfg, logger)

    with logger.timer(
        phase="ablation",
        step="run",
        component="spectramind.ablate",
        git_commit=_git_sha(),
        config_hash=_config_hash(cfg),
        cli_cmd=cmd,
    ):
        console.print("Ablation stub - supply overrides like model.fusion.type=concat")
        time.sleep(0.05)


if __name__ == "__main__":
    app()