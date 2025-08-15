from __future__ import annotations

import torch


def gaussian_log_likelihood(
    mu_pred: torch.Tensor, sigma_pred: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """Gaussian log-likelihood per bin."""
    var = sigma_pred**2 + 1e-12
    return 0.5 * (torch.log(2 * torch.pi * var) + (y_true - mu_pred) ** 2 / var)


def gll_mean(
    mu_pred: torch.Tensor, sigma_pred: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """Mean GLL across all bins and batch items."""
    return gaussian_log_likelihood(mu_pred, sigma_pred, y_true).mean()
