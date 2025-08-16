from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class GaussianLikelihoodLoss:
    """
    Negative log-likelihood for (mu, sigma) outputs vs. target y.
    Assumes sigma is a positive scale (we apply softplus if needed).
    """
    min_sigma: float = 1e-3
    clamp_sigma: Optional[float] = None  # If provided, clamp sigma <= this value

    def __call__(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(sigma) + self.min_sigma
        if self.clamp_sigma is not None:
            sigma = torch.clamp(sigma, max=self.clamp_sigma)

        # Gaussian NLL up to a constant: 0.5 * [ ((y-mu)/sigma)^2 + 2*log(sigma) ]
        z = (y - mu) / sigma
        return 0.5 * (z ** 2 + 2.0 * torch.log(sigma)).mean()


@dataclass
class SmoothnessLoss:
    """
    L2 smoothness penalty on adjacent bin differences (for spectral smoothness).
    """
    weight: float = 1.0
    dim: int = -1  # dimension along which bins lie

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        diffs = x.diff(dim=self.dim)
        return self.weight * (diffs ** 2).mean()


@dataclass
class AsymmetryLoss:
    """
    Penalize asymmetry across a midpoint (useful for physics-informed priors in some contexts).
    """
    weight: float = 1.0
    dim: int = -1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(self.dim)
        left = x.narrow(dim=self.dim, start=0, length=n // 2)
        right = x.narrow(dim=self.dim, start=n - left.size(self.dim), length=left.size(self.dim)).flip(self.dim)
        return self.weight * F.l1_loss(left, right)


class CompositeLoss(torch.nn.Module):
    """
    Composite loss wrapper merging a main loss (e.g., GLL) with optional auxiliary penalties.
    """
    def __init__(self, main_loss: GaussianLikelihoodLoss, smooth: Optional[SmoothnessLoss] = None, asym: Optional[AsymmetryLoss] = None, lambda_smooth: float = 0.0, lambda_asym: float = 0.0) -> None:
        super().__init__()
        self.main_loss = main_loss
        self.smooth = smooth
        self.asym = asym
        self.lambda_smooth = lambda_smooth
        self.lambda_asym = lambda_asym

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        gll = self.main_loss(mu, sigma, y)
        losses["gll"] = gll
        total = gll

        if self.smooth is not None:
            ls = self.smooth(mu)
            losses["smooth"] = ls
            total = total + self.lambda_smooth * ls

        if self.asym is not None:
            la = self.asym(mu)
            losses["asym"] = la
            total = total + self.lambda_asym * la

        losses["total"] = total
        return losses
