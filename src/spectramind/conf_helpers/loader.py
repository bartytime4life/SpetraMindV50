# SPDX-License-Identifier: MIT

"""High level configuration loading helpers."""

from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from omegaconf import DictConfig

from .hydra_integration import load_config_hydra
from .overrides import apply_symbolic_overrides, load_overrides_layered
from .validators import validate_config
from .hashing import config_hash
from .logging_utils import write_md, log_event


def load_and_validate(
    config_dir: str | Path,
    config_name: str = "config_v50.yaml",
    overrides: Optional[List[str]] = None,
    symbolic_override_path: Optional[str | Path] = None,
    layered_override_paths: Optional[List[str | Path]] = None,
) -> DictConfig:
    """Load a configuration, apply overrides, validate, and log."""
    cfg = load_config_hydra(config_dir, config_name, overrides)
    if layered_override_paths:
        cfg = load_overrides_layered(cfg, layered_override_paths)
    if symbolic_override_path:
        cfg = apply_symbolic_overrides(cfg, symbolic_override_path)
    _ = validate_config(cfg)
    h = config_hash(cfg)
    write_md("Loaded + validated config", {"hash": h})
    log_event("config_ready", {"hash": h})
    return cfg
