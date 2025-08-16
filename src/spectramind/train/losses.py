from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore


def gaussian_log_likelihood(mu, sigma, target, eps: float = 1e-6):
    """Compute per-sample Gaussian Log-Likelihood (negative log prob).

    Returns
    -------
    loss : tensor
        Mean NLL over batch and dimensions.
    """
    if torch is None:
        # Fallback: numpy-based approximation for environments without torch
        sigma_safe = np.clip(sigma, eps, None)
        nll = 0.5 * (
            np.log(2 * math.pi)
            + 2 * np.log(sigma_safe)
            + ((target - mu) ** 2) / (sigma_safe**2)
        )
        return float(np.mean(nll))

    sigma = torch.clamp(sigma, min=eps)
    nll = 0.5 * (
        math.log(2 * math.pi)
        + 2.0 * torch.log(sigma)
        + ((target - mu) ** 2) / (sigma**2)
    )
    return nll.mean()


def l2_smoothness(x, weight: float = 1.0):
    """Penalize second finite difference along the last dimension (spectral smoothness)."""
    if torch is None:
        d2 = np.diff(x, n=2, axis=-1)
        return float(weight) * float(np.mean(d2**2))
    d2 = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
    return weight * (d2**2).mean()


def fft_power_penalty(x, weight: float = 1.0, hi_freq_cut: Optional[int] = None):
    """Penalize high-frequency FFT power beyond ``hi_freq_cut``.

    Assumes last dimension is spectral/time.
    """
    if torch is None:
        X = np.fft.rfft(x, axis=-1)
        if hi_freq_cut is not None:
            X[..., :hi_freq_cut] = 0
        return float(weight) * float(np.mean(np.abs(X) ** 2))
    X = torch.fft.rfft(x, dim=-1)
    if hi_freq_cut is not None:
        X = X[..., hi_freq_cut:]
    return weight * (X.abs() ** 2).mean()


def symbolic_constraints(mu, cfg: Dict) -> Tuple[Optional["torch.Tensor"], Dict[str, float]]:
    """Hook for symbolic/physics-informed constraints. Non-fatal if unavailable.

    This function looks for a callable specified in cfg like::

        cfg.symbolic = {"callable": "spectramind.symbolic.symbolic_loss:compute", "weight": 0.1}

    It dynamically imports and runs the symbolic loss if available. Returns:
        (loss_tensor_or_None, metrics_dict)
    """
    weight = float(cfg.get("weight", 0.0)) if cfg else 0.0
    if weight <= 0:
        return None, {}

    target = None
    try:
        path = cfg.get("callable", "")
        if not path or ":" not in path:
            return None, {}
        mod_name, func_name = path.split(":", 1)
        mod = __import__(mod_name, fromlist=[func_name])
        func = getattr(mod, func_name)
        loss_tensor, info = func(mu=mu, cfg=cfg)  # type: ignore
        if loss_tensor is None:
            return None, {}
        if torch is None:
            return None, info or {}
        return weight * loss_tensor, info or {}
    except Exception:
        return None, {}


def assemble_losses(mu, sigma, target, cfg: Dict) -> Tuple["torch.Tensor" | float, Dict[str, float]]:
    """Compose all losses based on cfg dict."""
    metrics: Dict[str, float] = {}

    # Gaussian Log-Likelihood
    gll_w = float(cfg.get("gll", {}).get("weight", 1.0))
    gll = gaussian_log_likelihood(mu, sigma, target)
    total = gll_w * gll
    metrics["loss_gll"] = float(gll if isinstance(gll, (float, int)) else gll.item())
    metrics["w_gll"] = gll_w

    # Smoothness
    sm_w = float(cfg.get("smooth", {}).get("weight", 0.0))
    if sm_w > 0:
        sm = l2_smoothness(mu, weight=sm_w)
        total = total + sm
        metrics["loss_smooth"] = float(sm if isinstance(sm, (float, int)) else sm.item())
        metrics["w_smooth"] = sm_w

    # FFT
    fft_cfg = cfg.get("fft", {})
    fft_w = float(fft_cfg.get("weight", 0.0))
    if fft_w > 0:
        cutoff = fft_cfg.get("hi_freq_cut", None)
        fp = fft_power_penalty(mu, weight=fft_w, hi_freq_cut=cutoff)
        total = total + fp
        metrics["loss_fft"] = float(fp if isinstance(fp, (float, int)) else fp.item())
        metrics["w_fft"] = fft_w

    # Symbolic
    sym_loss, sym_info = symbolic_constraints(mu, cfg.get("symbolic", {}))
    if sym_loss is not None:
        total = total + sym_loss
        for k, v in (sym_info or {}).items():
            metrics[f"sym_{k}"] = float(v)

    return total, metrics

