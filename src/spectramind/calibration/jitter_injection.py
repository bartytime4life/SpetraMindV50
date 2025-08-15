"""Sub-pixel jitter injection for augmentation."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.ndimage import shift as imshift  # type: ignore
except Exception:  # pragma: no cover - fallback
    def imshift(arr: np.ndarray, shift, order=1, mode="nearest") -> np.ndarray:  # type: ignore
        if isinstance(shift, (tuple, list, np.ndarray)):
            s = [int(round(v)) for v in shift]
        else:
            s = [int(round(shift)), 0, 0]
        return np.roll(arr, shift=s, axis=(0, 1, 2))

from .calibration_config import JitterInjectionConfig


def inject_jitter(cube: np.ndarray, cfg: JitterInjectionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Inject random sub-pixel jitter along spatial axes."""

    if not cfg.enabled:
        return cube, np.zeros((cube.shape[0], 2), dtype=np.float32)

    rng = np.random.default_rng(cfg.seed)
    T = cube.shape[0]
    shifts = rng.normal(0.0, cfg.std_px, size=(T, 2)).astype(np.float32)
    out = np.empty_like(cube)
    for t in range(T):
        dy, dx = float(shifts[t, 0]), float(shifts[t, 1])
        out[t] = imshift(cube[t], shift=(dy, dx, 0), order=1, mode="nearest")
    return out, shifts
