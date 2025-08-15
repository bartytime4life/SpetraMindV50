# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for symbolic rules."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

# Logging / JSONL -----------------------------------------------------------


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def get_logger(
    name: str = "spectramind", logfile: Optional[str] = None
) -> logging.Logger:
    """Create/retrieve a Hydra-safe logger."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(sh)
    if logfile is not None and not any(
        isinstance(h, RotatingFileHandler)
        and h.baseFilename == os.path.abspath(logfile)
        for h in logger.handlers
    ):
        ensure_dir(logfile)
        fh = RotatingFileHandler(logfile, maxBytes=2_000_000, backupCount=5)
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(fh)
    return logger


def jsonl_event(path: Optional[str], record: Dict[str, Any]) -> None:
    if not path:
        return
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_git_revision(default: Optional[str] = None) -> Optional[str]:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return default


# Tensor helpers ------------------------------------------------------------


def ensure_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_tensor(x: Any, device: torch.device) -> Optional[Tensor]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def summarize_tensor(
    x: Optional[Tensor], name: str = "x", k: int = 5
) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    x_cpu = x.detach().flatten().float().cpu()
    if x_cpu.numel() == 0:
        return {"name": name, "numel": 0}
    vals = x_cpu
    return {
        "name": name,
        "numel": int(vals.numel()),
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
        "mean": float(vals.mean().item()),
        "std": float(vals.std().item()),
    }


# FFT utilities -------------------------------------------------------------


def rfft_power_spectrum(x: Tensor) -> Tensor:
    X = torch.fft.rfft(x, dim=1, norm="forward")
    return X.real.pow(2) + X.imag.pow(2)


def fft_rfft_power(x: Tensor, tail_frac: float = 0.25) -> Tensor:
    P = rfft_power_spectrum(x)
    F = P.size(1)
    tail = max(1, int(tail_frac * F))
    tail_power = P[:, -tail:].sum(dim=1)
    total = P.sum(dim=1).clamp_min(1e-12)
    return tail_power / total


# Wavelength helpers -------------------------------------------------------


def get_wavelengths(
    metadata: Optional[Dict[str, Any]], N: int, device: torch.device
) -> Tensor:
    if isinstance(metadata, dict) and metadata.get("wavelengths") is not None:
        wl = metadata["wavelengths"]
        if not torch.is_tensor(wl):
            wl = torch.as_tensor(wl, dtype=torch.float32, device=device)
        wl = wl.to(device=device, dtype=torch.float32)
        if wl.dim() == 1:
            wl = wl.unsqueeze(0)
        if wl.size(-1) != N:
            wl = torch.linspace(1.0, float(N), N, device=device).unsqueeze(0)
        return wl
    return torch.linspace(0.5, 7.8, N, device=device).unsqueeze(0)


def build_band_masks(
    wavelengths: Tensor,
    molecules: Dict[str, List[float]],
    band_half_width: float = 0.05,
) -> Dict[str, Tensor]:
    B, N = wavelengths.size(0), wavelengths.size(1)
    masks: Dict[str, Tensor] = {}
    for mol, centers in molecules.items():
        m = torch.zeros((B, N), device=wavelengths.device, dtype=torch.float32)
        for c in centers:
            left = c - band_half_width
            right = c + band_half_width
            m = torch.maximum(
                m, ((wavelengths >= left) & (wavelengths <= right)).float()
            )
        masks[mol] = m
    return masks


def build_centers_mask(
    wavelengths: Tensor, center_um: float, window_um: float
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    B = wavelengths.size(0)
    mask = (
        (wavelengths >= (center_um - window_um))
        & (wavelengths <= (center_um + window_um))
    ).float()
    if mask.sum() == 0:
        return None, None, None
    left_idx = torch.zeros(B, dtype=torch.long, device=wavelengths.device)
    right_idx = torch.zeros(B, dtype=torch.long, device=wavelengths.device)
    for b in range(B):
        idx = torch.nonzero(mask[b] > 0.5, as_tuple=False).flatten()
        if idx.numel() == 0:
            left_idx[b] = 0
            right_idx[b] = -1
        else:
            left_idx[b] = int(idx.min().item())
            right_idx[b] = int(idx.max().item())
    return mask, left_idx, right_idx


def build_optional_region_mask(
    metadata: Optional[Dict[str, Any]], N: int, region: Dict[str, float]
) -> Optional[Tensor]:
    if not region or metadata is None:
        return None
    wl = get_wavelengths(metadata, N, device=torch.device("cpu"))
    min_um = region.get("min_um")
    max_um = region.get("max_um")
    mask = torch.ones_like(wl)
    if min_um is not None:
        mask = mask * (wl >= min_um).float()
    if max_um is not None:
        mask = mask * (wl <= max_um).float()
    if (
        isinstance(metadata.get("wavelengths"), torch.Tensor)
        and metadata["wavelengths"].dim() == 2
    ):
        return mask
    return mask


def robust_baseline(mu: Tensor, out_of_band_mask: Tensor) -> Tensor:
    B, N = mu.shape
    baseline = []
    for b in range(B):
        mask = out_of_band_mask[b] > 0.5
        vals = mu[b, mask] if mask.any() else mu[b]
        med = torch.median(vals)
        baseline.append(med)
    return torch.stack(baseline, dim=0).unsqueeze(1)


def normalize_zero_one(x: Tensor, eps: float = 1e-6) -> Tensor:
    minv = x.amin(dim=1, keepdim=True)
    maxv = x.amax(dim=1, keepdim=True)
    return (x - minv) / (maxv - minv + eps)


def safe_corrcoef(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    if x.dim() == 2:
        x = x.mean(dim=1)
    if y.dim() == 2:
        y = y.mean(dim=1)
    xm = x - x.mean(dim=0, keepdim=True)
    ym = y - y.mean(dim=0, keepdim=True)
    num = (xm * ym).sum(dim=0)
    den = torch.sqrt((xm.pow(2).sum(dim=0) + eps) * (ym.pow(2).sum(dim=0) + eps))
    return (num / den).clamp(min=-1.0, max=1.0)


__all__ = [
    "ensure_dir",
    "get_logger",
    "jsonl_event",
    "now_utc_iso",
    "get_git_revision",
    "ensure_device",
    "to_tensor",
    "summarize_tensor",
    "rfft_power_spectrum",
    "fft_rfft_power",
    "get_wavelengths",
    "build_band_masks",
    "build_centers_mask",
    "build_optional_region_mask",
    "robust_baseline",
    "normalize_zero_one",
    "safe_corrcoef",
]
