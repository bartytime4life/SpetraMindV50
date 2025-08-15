"""Flat-field correction."""

from __future__ import annotations

import numpy as np

from .calibration_config import FlatConfig


def apply_flat_field(cube: np.ndarray, cfg: FlatConfig) -> np.ndarray:
    """Apply flat-field correction by dividing by a flat frame or median estimate."""

    if not cfg.enabled:
        return cube
    if cfg.frame_path is None:
        flat = np.median(cube, axis=0)
    else:
        flat = np.load(cfg.frame_path)
    denom = np.where(np.abs(flat) < cfg.eps, cfg.eps, flat)
    return cube / denom
