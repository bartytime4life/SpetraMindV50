import datetime
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .diagnostics_hooks import DiagnosticsHooks
from .event_stream import EventStream
from .metrics_logger import MetricsLogger


def _git_info() -> Dict[str, Any]:
    """Best-effort capture of git hash and branch for reproducibility."""

    def _run(cmd: list[str]) -> Optional[str]:
        try:
            return (
                subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            )
        except Exception:
            return None

    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status_dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _env_info() -> Dict[str, Any]:
    """Capture a small, non-sensitive ENV snapshot (tooling & python)."""

    return {
        "python_executable": sys.executable,
        "python_version": os.environ.get("PYTHON_VERSION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "spectramind_config_hash": os.environ.get("SPECTRAMIND_CONFIG_HASH"),
        "spectramind_cli_version": os.environ.get("SPECTRAMIND_CLI_VERSION"),
    }


class TelemetryManager:
    """
    Master telemetry manager for SpectraMind V50.

    Responsibilities:
    - Initialize rotating logs, JSONL event stream, and metrics sinks.
    - Synchronize telemetry with diagnostics dashboard and MLflow/W&B (optional).
    - Provide mission-grade reproducibility by embedding Git hash, ENV, and config hash.
    """

    def __init__(
        self,
        log_dir: str = "logs/telemetry",
        enable_jsonl: bool = True,
        sync_mlflow: bool = False,
        sync_wandb: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"telemetry_{timestamp}.log"
        self.jsonl_file = self.log_dir / f"telemetry_{timestamp}.jsonl"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(),
            ],
        )

        self.event_stream = EventStream(self.jsonl_file if enable_jsonl else None)
        self.metrics_logger = MetricsLogger(self.log_dir / "metrics.csv")
        self.diagnostics_hooks = DiagnosticsHooks()
        self.sync_mlflow = sync_mlflow
        self.sync_wandb = sync_wandb

        session_info = {
            "session_start": datetime.datetime.utcnow().isoformat(),
            "git": _git_info(),
            "env": _env_info(),
        }
        self.log_event("telemetry_session_start", session_info)
        logging.info("TelemetryManager initialized")

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Log a structured telemetry event to JSONL and console."""
        event = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        self.event_stream.write_event(event)
        logging.info(f"[EVENT] {event_type}: {payload}")

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a numeric metric with timestamp and step."""
        self.metrics_logger.log_metric(name, value, step)

    def attach_diagnostics(self, diagnostics_fn: Callable[[], Dict[str, Any]]) -> None:
        """Attach a custom diagnostics hook (called periodically)."""
        self.diagnostics_hooks.register_hook(diagnostics_fn)

    def run_diagnostics(self) -> Dict[str, Any]:
        """Run all registered diagnostic hooks and log results."""
        results = self.diagnostics_hooks.run_hooks()
        self.log_event("diagnostics", results)
        return results
