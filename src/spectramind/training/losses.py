from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Bin-wise Gaussian negative log-likelihood. sigma is standard deviation (>0).
    Returns mean over batch and bins.
    """
    sigma = torch.clamp(sigma, min=eps)
    var = sigma * sigma
    nll = 0.5 * (torch.log(2 * torch.pi * var) + (mu - target) ** 2 / var)
    return nll.mean()


def smoothness_l2(mu: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """
    L2 smoothness across spectral bins: sum of squared first differences.
    Expects shape [B, BINS] or [B, BINS, ...] (diff along last-1 axis).
    """
    if mu.ndim < 2:
        return mu.new_tensor(0.0)
    diff = mu[..., 1:] - mu[..., :-1]
    return lam * (diff.pow(2).mean())


def asymmetry_penalty(mu: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """
    Simple asymmetry penalty via signed second derivative preference towards symmetry.
    """
    if mu.ndim < 2 or mu.size(-1) < 3:
        return mu.new_tensor(0.0)
    d1 = mu[..., 1:] - mu[..., :-1]
    d2 = d1[..., 1:] - d1[..., :-1]
    # Penalize large signed curvature (encourage symmetry)
    return lam * d2.abs().mean()


def maybe_symbolic_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    meta: Optional[Dict[str, Any]],
    weight: float = 0.0,
) -> torch.Tensor:
    """
    Optionally call SpectraMind symbolic loss if available.
    This function fails soft (returns zero) if the module is not present.
    """
    if weight <= 0:
        if outputs["mu"].is_cuda:
            return outputs["mu"].new_tensor(0.0, device=outputs["mu"].device)
        return outputs["mu"].new_tensor(0.0)
    try:
        from spectramind.symbolic.symbolic_loss import compute_symbolic_loss  # type: ignore
    except Exception:
        logging.warning("Symbolic loss module not found; continuing without it.")
        return outputs["mu"].new_tensor(0.0, device=outputs["mu"].device)
    return weight * compute_symbolic_loss(outputs=outputs, target=target, meta=meta)


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    meta: Optional[Dict[str, Any]],
    cfg_loss: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Compute composite loss from configured components.
    cfg_loss example:
    {
    "gll": {"weight": 1.0},
    "smooth": {"weight": 0.02},
    "asym": {"weight": 0.01},
    "symbolic": {"weight": 0.5}
    }
    """
    mu = outputs["mu"]
    sigma = outputs.get("sigma", None)
    if sigma is None:
        # If model doesn't output sigma, use a fixed sigma=1 as fallback
        sigma = mu.new_ones(mu.shape)

    weights = {k: float(v.get("weight", 0.0)) for k, v in cfg_loss.items()}
    losses: Dict[str, torch.Tensor] = {}

    gll = gaussian_nll(mu, sigma, target) * weights.get("gll", 1.0)
    losses["gll"] = gll

    if weights.get("smooth", 0.0) > 0:
        losses["smooth"] = smoothness_l2(mu, lam=weights["smooth"])
    else:
        losses["smooth"] = mu.new_tensor(0.0)

    if weights.get("asym", 0.0) > 0:
        losses["asym"] = asymmetry_penalty(mu, lam=weights["asym"])
    else:
        losses["asym"] = mu.new_tensor(0.0)

    if weights.get("symbolic", 0.0) > 0:
        losses["symbolic"] = maybe_symbolic_loss(outputs, target, meta, weight=weights["symbolic"])
    else:
        losses["symbolic"] = mu.new_tensor(0.0)

    losses["total"] = sum(losses.values())
    return losses
