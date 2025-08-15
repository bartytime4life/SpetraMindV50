"""Light-curve normalisation routines."""

from __future__ import annotations

import numpy as np

from .calibration_config import NormalizationConfig


def phase_normalize(flux: np.ndarray, cfg: NormalizationConfig) -> tuple[np.ndarray, dict]:
    """Normalize light curve to ~1 outside transit."""

    if not cfg.enabled:
        return flux, {"method": "none"}

    f = flux.astype(np.float64)
    if cfg.method == "median-oot":
        ql, qu = np.quantile(f, [0.1, 0.9])
        oot_mask = (f >= ql) & (f <= qu)
        baseline = np.median(f[oot_mask])
        norm = f if baseline == 0.0 else f / baseline
        return norm, {
            "method": "median-oot",
            "baseline": float(baseline),
            "oot_count": int(oot_mask.sum()),
        }

    x = np.linspace(-1, 1, f.shape[0])
    A = np.vstack([x ** k for k in range(cfg.poly_order + 1)]).T
    w = np.ones_like(f)
    for _ in range(3):
        coeff, *_ = np.linalg.lstsq(A * w[:, None], f * w, rcond=None)
        base = A @ coeff
        resid = f - base
        s = 1.4826 * np.median(np.abs(resid)) + 1e-9
        w = (np.abs(resid) < 3 * s).astype(np.float64)
        base = A @ coeff
    norm = f if np.all(base == 0.0) else f / base
    return norm, {"method": "poly", "coeff": coeff.tolist()}
