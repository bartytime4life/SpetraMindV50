"""
Validator for Hydra configs using schema.py.
"""

import yaml
from pathlib import Path
from .schema import (
    ModelConfig,
    TrainingConfig,
    CalibrationConfig,
    DiagnosticsConfig,
    LoggingConfig,
)


def validate_config(config_path: Path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Dispatch validation
    if "fgs1_encoder" in cfg:
        return ModelConfig(**cfg)
    if "optimizer" in cfg:
        return TrainingConfig(**cfg)
    if "temperature_scaling" in cfg:
        return CalibrationConfig(**cfg)
    if "fft" in cfg:
        return DiagnosticsConfig(**cfg)
    if "console" in cfg:
        return LoggingConfig(**cfg)

    raise ValueError(f"Unknown config schema: {config_path}")
