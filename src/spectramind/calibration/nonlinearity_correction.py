"""Polynomial non-linearity correction."""

from __future__ import annotations

import numpy as np

from .calibration_config import NonlinearityConfig


def apply_nonlinearity_correction(cube: np.ndarray, cfg: NonlinearityConfig) -> np.ndarray:
    """Apply a polynomial correction with a single Newton step approximation."""

    if not cfg.enabled:
        return cube

    coeffs = np.array(cfg.coeffs, dtype=np.float32)
    y = np.polyval(coeffs[::-1], cube)

    dcoeffs = np.polyder(coeffs[::-1])
    dy = np.polyval(dcoeffs, y)
    dy = np.where(np.abs(dy) < 1e-6, 1.0, dy)
    x_est = y / dy
    return x_est.astype(cube.dtype, copy=False)
