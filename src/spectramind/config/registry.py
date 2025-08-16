"""
Registry for loading Hydra configs with validation.
"""

from pathlib import Path

from hydra import compose, initialize

from .schema import (
    ModelConfig,
    TrainingConfig,
    CalibrationConfig,
    DiagnosticsConfig,
    LoggingConfig,
)

CONFIG_PATH = Path(__file__).resolve().parent


def load_config(config_name: str, overrides: list | None = None):
    overrides = overrides or []
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name=config_name, overrides=overrides)

    # Validate
    if config_name == "model":
        ModelConfig(**cfg)
    elif config_name == "training":
        TrainingConfig(**cfg)
    elif config_name == "calibration":
        CalibrationConfig(**cfg)
    elif config_name == "diagnostics":
        DiagnosticsConfig(**cfg)
    elif config_name == "logging":
        LoggingConfig(**cfg)

    return cfg
