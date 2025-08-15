"""
Copyright (c) 2025 SpectraMind.

SPDX-License-Identifier: Apache-2.0

SpectraMind V50 Config Package

This package centralizes configuration dataclasses, Hydra loading, validation, hashing,
and environment/Git capture for the NeurIPS 2025 Ariel Data Challenge pipeline.

Design goals:
    •   Science-first, reproducible defaults with explicit typing and validation.
    •   Hydra-compatible composition and CLI override safety.
    •   Mission-aware defaults (Ariel FGS1/AIRS) baked into config groups.
    •   Logging-friendly utilities (JSONL event stream, rotating logs path scaffolding).
    •   Git/ENV snapshot for run fingerprints and MLflow/W&B sync toggles.

Exports:
    •   Config dataclasses (Config, ModelConfig, TrainingConfig, DataConfig, PathsConfig, LoggingConfig)
    •   load_hydra_config() for robust on-disk discovery of the configs/ tree
    •   validate_config() for schema + cross-field checks
    •   hash_config(), save_config(), load_config() for run immutability
    •   capture_env_git_snapshot() to embed run metadata

Notes:
    •   Keep this package import-only (no top-level Hydra initialization).
    •   Avoid any device/framework imports here to stay lightweight at CLI import time.
"""

from .base_config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    PathsConfig,
    LoggingConfig,
    Config,
)

from .validators import validate_config
from .hydra_loader import load_hydra_config
from .utils import hash_config, save_config, load_config
from .env_capture import capture_env_git_snapshot

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "PathsConfig",
    "LoggingConfig",
    "Config",
    "validate_config",
    "load_hydra_config",
    "hash_config",
    "save_config",
    "load_config",
    "capture_env_git_snapshot",
]

__version__ = "0.1.0"
