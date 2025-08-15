"""Photometric extraction routines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .calibration_config import PhotometryConfig


def _circular_mask(h: int, w: int, cx: float, cy: float, r: float) -> NDArray[np.bool_]:
    y, x = np.ogrid[:h, :w]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def aperture_photometry(
    cube: NDArray[np.floating], cfg: PhotometryConfig
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Simple circular aperture photometry with background annulus."""

    T, H, W = cube.shape
    cx, cy = W / 2.0, H / 2.0
    r = cfg.aperture_radius_px
    mask_ap = _circular_mask(H, W, cx, cy, r)
    mask_bg = _circular_mask(H, W, cx, cy, cfg.background_outer_px) & (~_circular_mask(H, W, cx, cy, cfg.background_inner_px))
    ap_area = np.clip(mask_ap.sum(), 1, None)
    bg_area = np.clip(mask_bg.sum(), 1, None)

    flux = np.empty(T, dtype=np.float64)
    err = np.empty(T, dtype=np.float64)
    for t in range(T):
        frame = cube[t]
        bg = frame[mask_bg].mean() if bg_area > 0 else 0.0
        signal = frame[mask_ap].sum() - bg * ap_area
        flux[t] = signal
        err[t] = np.sqrt(np.maximum(frame[mask_ap].sum(), 1.0))
    return flux, err


def optimal_photometry(
    cube: NDArray[np.floating], cfg: PhotometryConfig
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Toy optimal photometry weighted by a Gaussian PSF."""

    T, H, W = cube.shape
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0
    sigma = cfg.aperture_radius_px / 2.0
    psf = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))
    psf /= psf.sum() + 1e-12
    flux = (cube * psf).reshape(T, -1).sum(axis=1)
    err = np.sqrt(np.maximum(cube.reshape(T, -1).sum(axis=1), 1.0))
    return flux, err


def extract_photometry(
    cube: NDArray[np.floating], cfg: PhotometryConfig
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Dispatch photometric extraction method."""

    if cfg.method == "aperture":
        return aperture_photometry(cube, cfg)
    return optimal_photometry(cube, cfg)
