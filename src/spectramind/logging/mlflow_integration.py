# SPDX-License-Identifier: MIT
"""Optional MLflow integration for SpectraMind logging."""
from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import mlflow  # type: ignore
    _MLFLOW_OK = True
except Exception:  # pragma: no cover - MLflow not installed
    mlflow = None  # type: ignore
    _MLFLOW_OK = False

from .config import LoggingConfig


def mlflow_available() -> bool:
    """Return True if MLflow is importable."""
    return _MLFLOW_OK


class MLflowLogger:
    """Thin wrapper around MLflow logging functions."""

    def __init__(self, cfg: LoggingConfig):
        self.cfg = cfg
        self._active = bool(cfg.mlflow and _MLFLOW_OK)
        self._run = None
        if self._active:
            mlflow.set_experiment(cfg.experiment)  # type: ignore[attr-defined]
            self._run = mlflow.start_run(run_name=cfg.run_name)  # type: ignore[attr-defined]

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self._active:
            mlflow.log_metric(key, value, step=step)  # type: ignore[attr-defined]

    def log_param(self, key: str, value: Any) -> None:
        if self._active:
            mlflow.log_param(key, value)  # type: ignore[attr-defined]

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if self._active:
            mlflow.set_tags(tags)  # type: ignore[attr-defined]

    def finish(self) -> None:
        if self._active:
            mlflow.end_run()  # type: ignore[attr-defined]
            self._active = False
            self._run = None
