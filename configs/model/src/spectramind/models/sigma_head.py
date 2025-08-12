from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmaHeadFlow(nn.Module):
    """
    Positive sigma head. Stub for flow-style; implemented as softplus MLP.
    Input: (B, N, D) -> Output: (B, N)
    """
    def __init__(self, in_dim: int, latent_dim: int, dropout: float = 0.05, sigma_min: float = 1.0e-4):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)      # (B, N)
        sigma = F.softplus(raw) + self.sigma_min
        return sigma