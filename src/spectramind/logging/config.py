"""
Hydra-Safe Logging Configuration
--------------------------------
Defines structured config for logging, reproducibility, and telemetry.
"""
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_dir: str = "logs"
    console: bool = True
    file: bool = True
    file_max_mb: int = 50
    file_backup_count: int = 5
    jsonl: bool = True
    mlflow: bool = False
    wandb: bool = False
    project: str = "spectramind-v50"
    run_name: str = "default"
    experiment: str = "main"
