"""
YAML/JSON config loader with Hydra/OmegaConf support and schema validation.

Features:
- load_config(path): Load YAML/JSON and validate against config_schema.yaml
- load_hydra_config(overrides): Create an OmegaConf from dict overrides (Hydra-style), validate, and return dict
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .config_validator import validate_config

def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML or JSON configuration file and validate it.

    Args:
        path: Path to a .yaml/.yml or .json file.

    Returns:
        Dict[str, Any]: The validated configuration dictionary.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If file extension unsupported or schema validation fails.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path_obj, "r") as f:
        if path_obj.suffix in (".yaml", ".yml"):
            cfg = yaml.safe_load(f) or {}
        elif path_obj.suffix == ".json":
            cfg = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path_obj.suffix}")
    validate_config(cfg)
    return cfg

def load_hydra_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an OmegaConf/Hydra-style configuration from a dict of overrides, validate it, and return as a plain dict.

    Note:
      - This does NOT launch a Hydra job; it only builds an in-memory config for convenience.
      - If Hydra/OmegaConf are not installed, import will fail; install via `pip install hydra-core omegaconf`.

    Args:
        overrides: Dict[str, Any] of nested keys/values (e.g., {"training": {"epochs": 100}})

    Returns:
        Dict[str, Any]: Resolved plain dict after validation.
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(overrides or {})
    resolved = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    validate_config(resolved)  # type: ignore[arg-type]
    return resolved  # type: ignore[return-value]
