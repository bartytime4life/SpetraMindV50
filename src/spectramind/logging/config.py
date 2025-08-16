# SPDX-License-Identifier: MIT
"""SpectraMind V50 logging configuration.

Defines a Hydra/OmegaConf friendly dataclass controlling console and file
handlers, JSONL event streams and optional external integrations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class LoggingConfig:
    """Dataclass configuration for the logging stack."""

    # Core behaviour
    log_level: str = "INFO"  # DEBUG|INFO|WARNING|ERROR
    log_dir: str = "logs"  # Directory for spectramind.log and events.jsonl
    console: bool = True  # Enable console handler
    file: bool = True  # Enable rotating file handler
    file_max_mb: int = 32  # Max log file size before rotation
    file_backup_count: int = 3  # Number of rotated backups to keep
    jsonl: bool = True  # Enable JSONL event stream
    jsonl_indent: Optional[int] = None  # Indent for JSONL (None for compact)

    # Optional integrations
    mlflow: bool = False  # Enable MLflow logging (if installed)
    wandb: bool = False  # Enable Weights & Biases logging (if installed)
    project: str = "spectramind-v50"  # Project name for external sinks
    run_name: str = "default"  # Run name/id for external sinks
    experiment: str = "main"  # Experiment name for MLflow

    # Formatting overrides (advanced)
    console_fmt: str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    file_fmt: str = (
        "[%(asctime)s] [%(levelname)s] [%(name)s] [pid=%(process)d "
        "tid=%(threadName)s run=%(run_id)s git=%(git_hash)s] %(message)s"
    )
    date_fmt: str = "%Y-%m-%d %H:%M:%S"

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of this config."""
        return asdict(self)
