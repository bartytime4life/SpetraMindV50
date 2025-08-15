"""Dark current correction utilities."""

from __future__ import annotations

import numpy as np

from .calibration_config import DarkConfig


def apply_dark_correction(cube: np.ndarray, exposure_s: float, cfg: DarkConfig) -> np.ndarray:
    """Apply dark current correction."""

    if not cfg.enabled:
        return cube
    if cfg.method == "frame" and cfg.frame_path:
        dark = np.load(cfg.frame_path)
        return cube - dark
    return cube - (cfg.dark_rate_e_per_s * exposure_s)
