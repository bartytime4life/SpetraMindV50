"""Conformal risk calibration for uncertainty estimates."""

from __future__ import annotations

import numpy as np


def conformal_calibrate_sigma(
    mu_pred: np.ndarray,
    mu_true: np.ndarray,
    sigma_pred: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """Perform a simple bin-wise conformal calibration of sigma."""

    r = np.abs(mu_pred - mu_true)
    if r.ndim == 2:
        q_alpha = np.quantile(r, 1 - alpha, axis=0)
        sigma_ref = np.maximum(
            sigma_pred.mean(axis=0) if sigma_pred.ndim == 2 else sigma_pred, 1e-9
        )
    else:
        q_alpha = float(np.quantile(r, 1 - alpha))
        sigma_ref = np.maximum(np.mean(sigma_pred), 1e-9)

    scale = q_alpha / sigma_ref
    sigma_cal = sigma_pred * scale
    meta = {
        "alpha": alpha,
        "q_alpha": q_alpha.tolist() if hasattr(q_alpha, "tolist") else float(q_alpha),
        "scale_mean": float(np.mean(scale)),
    }
    return sigma_cal, meta
