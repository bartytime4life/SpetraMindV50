from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from hydra import compose, initialize
from omegaconf import OmegaConf


def resolve_config_source(
    config_path_or_name: str,
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Resolve a config path or hydra-style name.

    Returns:
        (config_dir, config_name_without_ext, is_hydra_style)
        - If is_hydra_style=True: use Hydra compose with (config_dir, config_name)
        - If False: treat as a plain YAML file path
    """
    p = Path(config_path_or_name)
    if p.exists() and p.is_file():
        return None, None, False
    # Heuristic: string like "config_v50.yaml" or "config_v50" in a known configs dir isn't guaranteed.
    # We assume hydra-style when it's not an existing file path.
    stem = p.stem
    return os.path.dirname(config_path_or_name) or ".", stem, True


def load_yaml(yaml_path: str):
    """Load a YAML file into an OmegaConf object."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return OmegaConf.create(data)


def load_config(config_path_or_name: str, overrides: Optional[List[str]] = None):
    """
    Load a config from either:
      - a direct YAML file path, or
      - a Hydra config directory + name (hydra-style)

    Args:
        config_path_or_name: YAML path or hydra name
        overrides: list of hydra override strings (e.g., ["train.lr=3e-4", "model=fgs1_mamba"])

    Returns:
        OmegaConf
    """
    overrides = overrides or []
    config_dir, config_name, is_hydra = resolve_config_source(config_path_or_name)
    if not is_hydra:
        return load_yaml(config_path_or_name)
    cfg_dir = config_dir
    cfg_name = config_name.replace(".yaml", "")
    with initialize(version_base=None, config_path=cfg_dir):
        cfg = compose(config_name=cfg_name, overrides=overrides)
    return cfg


def save_config(cfg, out_path: str):
    """Save an OmegaConf config to YAML."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)


def get_hydra_compose(
    config_dir: str, config_name: str, overrides: Optional[List[str]] = None
):
    """Convenience wrapper around Hydra compose."""
    overrides = overrides or []
    with initialize(version_base=None, config_path=config_dir):
        return compose(config_name=config_name, overrides=overrides)
