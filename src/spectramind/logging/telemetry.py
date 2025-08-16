# SPDX-License-Identifier: MIT
"""Lightweight telemetry helper built atop the logging package."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .mlflow_integration import MLflowLogger
from .wandb_integration import WandBLogger


class TelemetryLogger:
    """Emit telemetry events, metrics and params to logger/MLflow/W&B."""

    def __init__(
        self,
        logger: logging.Logger,
        mlflow_logger: Optional[MLflowLogger] = None,
        wandb_logger: Optional[WandBLogger] = None,
    ) -> None:
        self.log = logger
        self.mlflow = mlflow_logger
        self.wandb = wandb_logger

    def event(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "event": name,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.log.info("TelemetryEvent", extra={"telemetry": payload})
        if self.wandb:
            self.wandb.log({"event": name, **(metadata or {})})
        if self.mlflow:
            self.mlflow.log_param(f"event_{name}", payload["timestamp"])

    def metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        self.log.info(
            "TelemetryMetric",
            extra={"metric_key": key, "metric_value": value, "step": step},
        )
        if self.mlflow:
            self.mlflow.log_metric(key, value, step=step)
        if self.wandb:
            data = {key: value}
            if step is not None:
                data["_step"] = step
            self.wandb.log(data, step=step)

    def param(self, key: str, value: Any) -> None:
        self.log.info("TelemetryParam", extra={"param_key": key, "param_value": value})
        if self.mlflow:
            self.mlflow.log_param(key, value)
