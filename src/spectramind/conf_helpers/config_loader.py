"""Basic YAML config loading and saving utilities."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML configuration file into an ``OmegaConf`` ``DictConfig``."""
    return OmegaConf.load(Path(path))


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save a ``DictConfig`` to a YAML file."""
    OmegaConf.save(cfg, Path(path))
