# SPDX-License-Identifier: MIT
"""SpectraMind V50 logging package.

Unified mission-grade logging with console and rotating file handlers,
JSONL event streams, optional MLflow/W&B integrations, telemetry helpers,
and CI/self-test validators.
"""
from .config import LoggingConfig
from .logger import (
    init_logging,
    get_logger,
    add_context,
    get_git_hash,
    get_run_id,
    log_cli_call,
)
from .jsonl_handler import JSONLHandler
from .mlflow_integration import MLflowLogger, mlflow_available
from .wandb_integration import WandBLogger, wandb_available
from .telemetry import TelemetryLogger
from .validators import validate_logging_integrity

__all__ = [
    "LoggingConfig",
    "init_logging",
    "get_logger",
    "add_context",
    "get_git_hash",
    "get_run_id",
    "log_cli_call",
    "JSONLHandler",
    "MLflowLogger",
    "WandBLogger",
    "mlflow_available",
    "wandb_available",
    "TelemetryLogger",
    "validate_logging_integrity",
]
