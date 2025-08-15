"""Spectral trace alignment utilities."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.ndimage import shift as imshift  # type: ignore
except Exception:  # pragma: no cover - fallback

    def imshift(arr: np.ndarray, shift, order=1, mode="nearest") -> np.ndarray:  # type: ignore
        """Fallback implementation using ``np.roll`` when SciPy is unavailable."""

        if isinstance(shift, (tuple, list, np.ndarray)):
            s = [int(round(v)) for v in shift]
        else:
            s = [int(round(shift))]
        if len(s) < arr.ndim:
            s += [0] * (arr.ndim - len(s))
        return np.roll(arr, shift=tuple(s[: arr.ndim]), axis=tuple(range(arr.ndim)))


from .calibration_config import AlignmentConfig


def estimate_shift_1d(a: np.ndarray, b: np.ndarray, max_shift: int) -> float:
    """Estimate shift between 1D arrays using cross-correlation."""

    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    best_s, best_c = 0, -1e9
    for s in range(-max_shift, max_shift + 1):
        c = np.dot(
            a[max(0, s) : len(a) + min(0, s)], b[max(0, -s) : len(b) + min(0, -s)]
        )
        if c > best_c:
            best_c, best_s = c, s
    return float(best_s)


def align_spectral_traces(
    cube: np.ndarray, cfg: AlignmentConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Align along the spectral axis (assumed to be axis 1 for each frame)."""

    if not cfg.enabled:
        return cube, np.zeros(cube.shape[0], dtype=np.float32)

    T, H, W = cube.shape
    ref = cube[0]
    shifts = np.zeros(T, dtype=np.float32)
    out = np.empty_like(cube)
    out[0] = ref
    for t in range(1, T):
        s = estimate_shift_1d(ref.mean(axis=0), cube[t].mean(axis=0), cfg.max_shift_px)
        shifts[t] = s
        if cfg.subpixel:
            out[t] = imshift(cube[t], shift=(0, -s), order=1, mode="nearest")
        else:
            out[t] = np.roll(cube[t], shift=int(-s), axis=1)
    return out, shifts
