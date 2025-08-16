from __future__ import annotations

from typing import Dict

import torch


def rmse(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((mu - target) ** 2))


def mae(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(mu - target))


def coverage(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """
    Empirical coverage assuming mu ± z*sigma should contain target.
    Returns fraction of bins covered.
    """
    lower = mu - z * sigma
    upper = mu + z * sigma
    inside = (target >= lower) & (target <= upper)
    return inside.float().mean()


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sigma = torch.clamp(sigma, min=eps)
    var = sigma * sigma
    nll = 0.5 * (torch.log(2 * torch.pi * var) + (mu - target) ** 2 / var)
    return nll.mean()


def compute_metrics(outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, float]:
    """
    Compute core metrics used during training/validation.
    """
    mu = outputs["mu"]
    sigma = outputs.get("sigma", torch.ones_like(mu))
    with torch.no_grad():
        m = {
            "rmse": rmse(mu, target).item(),
            "mae": mae(mu, target).item(),
            "gll": gaussian_nll(mu, sigma, target).item(),
            "coverage68": coverage(mu, sigma, target, z=1.0).item(),
        }
        return m
