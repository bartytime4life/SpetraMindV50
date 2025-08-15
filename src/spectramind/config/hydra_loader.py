"""
Copyright (c) 2025 SpectraMind.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Any, Mapping

from omegaconf import OmegaConf
import hydra

from .base_config import Config


def _find_configs_dir(start: Optional[Path] = None) -> Path:
    """
    Walk upwards from start (or CWD) to locate a 'configs' directory.
    Returns the absolute Path to the directory, or raises FileNotFoundError.
    """
    here = Path(start or os.getcwd()).resolve()
    for p in [here, *here.parents]:
        candidate = p / "configs"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not locate 'configs' directory. Ensure repo layout has ./configs.")


def _to_container(cfg: Any) -> Mapping[str, Any]:
    """
    OmegaConf -> plain container with resolved interpolations.
    """
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def load_hydra_config(config_name: str = "config_v50", config_dir: Optional[str | Path] = None) -> Config:
    """
    Robust Hydra loader that auto-discovers the configs/ tree even when called from arbitrary CWD.

    Args:
        config_name: Top-level Hydra YAML to compose (without .yaml).
        config_dir: Optional path to configs directory; if None, auto-discover upwards.

    Returns:
        Config dataclass with extras capturing unknown keys.

    Usage:
        cfg = load_hydra_config("config_v50")
    """
    cfg_dir = Path(config_dir) if config_dir else _find_configs_dir()
    # Hydra requires relative path from the search root; use absolute with initialize to pin directory.
    with hydra.initialize(version_base=None, config_path=str(cfg_dir)):
        composed = hydra.compose(config_name=config_name)
        container = _to_container(composed)
        return Config.from_mapping(container)
