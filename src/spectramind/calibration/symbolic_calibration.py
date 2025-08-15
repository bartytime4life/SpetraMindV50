"""Symbolic calibration enforcing simple constraints."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.ndimage import uniform_filter1d  # type: ignore
except Exception:  # pragma: no cover - fallback
    def uniform_filter1d(a, size, axis, mode="nearest"):
        kernel = np.ones(size) / size
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis, a)

from .calibration_config import SymbolicCalibrationConfig


def apply_symbolic_constraints(
    cube: np.ndarray,
    cfg: SymbolicCalibrationConfig,
    axis_map: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply simple physics-informed constraints."""

    if not cfg.enabled:
        return cube, {"enabled": False}

    out = cube.copy()
    meta: dict[str, int | str] = {}

    if cfg.nonnegativity:
        neg_count = int((out < 0).sum())
        out = np.maximum(out, 0.0)
        meta["negatives_clipped"] = neg_count

    if cfg.smooth_window > 1:
        ax = 2
        if axis_map and cfg.smooth_axis in axis_map:
            ax = axis_map[cfg.smooth_axis]
        out = uniform_filter1d(out, size=cfg.smooth_window, axis=ax, mode="nearest")
        meta["smooth_window"] = cfg.smooth_window
        meta["smooth_axis"] = cfg.smooth_axis

    return out, meta
