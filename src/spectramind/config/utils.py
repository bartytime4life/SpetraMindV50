"""
Copyright (c) 2025 SpectraMind.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .base_config import Config, ModelConfig, TrainingConfig, DataConfig, PathsConfig, LoggingConfig


def hash_config(cfg: Config) -> str:
    """
    Compute a stable SHA256 hash over a canonical JSON serialization of the config.
    Use for run fingerprints, artifact naming, and CI reproducibility.
    """
    canonical = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_config(cfg: Config, path: str | Path) -> None:
    """
    Persist Config to disk as pretty-printed JSON; parent dirs are created as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)


def load_config(path: str | Path) -> Config:
    """
    Load Config from JSON saved by save_config(). Unknown keys are captured into extras.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    # Recompose dataclasses with defensive kwargs
    model = ModelConfig(**blob.get("model", {}))
    training = TrainingConfig(**blob.get("training", {}))
    data = DataConfig(**blob.get("data", {}))
    paths = PathsConfig(**blob.get("paths", {}))
    logging = LoggingConfig(**blob.get("logging", {}))
    extras = blob.get("extras", {})
    return Config(model=model, training=training, data=data, paths=paths, logging=logging, extras=extras)
