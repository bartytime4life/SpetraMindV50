"""Cosmic ray removal via temporal sigma clipping."""

from __future__ import annotations

import numpy as np

from .calibration_config import CosmicConfig


def remove_cosmics_sigma_clip(cube: np.ndarray, cfg: CosmicConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sigma-clip outliers in time for each pixel."""

    if not cfg.enabled:
        return cube, np.zeros_like(cube, dtype=np.uint8)

    T = cube.shape[0]
    median = np.median(cube, axis=0, keepdims=True)
    mad = np.median(np.abs(cube - median), axis=0, keepdims=True) + 1e-9
    z = (cube - median) / (1.4826 * mad)
    mask = (np.abs(z) > cfg.sigma).astype(np.uint8)

    clean = cube.copy()
    w = cfg.time_window
    for t in range(T):
        bad = mask[t] > 0
        if not bad.any():
            continue
        t0, t1 = max(0, t - w), min(T, t + w + 1)
        neigh = cube[t0:t1]
        rep = np.median(neigh, axis=0)
        clean[t][bad] = rep[bad]
    return clean, mask
