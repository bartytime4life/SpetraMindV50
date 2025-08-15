"""Temperature scaling utilities."""

from __future__ import annotations

import numpy as np


def fit_temperature_scaling(residuals: np.ndarray, sigma_pred: np.ndarray) -> float:
    """Fit global temperature scaling parameter T."""

    std_r = float(np.std(residuals))
    mean_s = float(np.mean(np.abs(sigma_pred)) + 1e-12)
    return std_r / mean_s


def apply_temperature_scaling(sigma_pred: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to predicted sigma."""

    return sigma_pred * T
