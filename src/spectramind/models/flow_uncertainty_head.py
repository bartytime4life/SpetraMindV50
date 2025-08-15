# σ (uncertainty) head: lightweight parametric head with temperature scaling
# and optional monotone smoothing prior via 1D conv.

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class FlowUncertaintyHead(nn.Module):
    """
    Returns
    -------
    mu  : [B, out_bins]
    sigma: [B, out_bins] strictly positive via softplus + eps
    """
    def __init__(self, in_dim: int = 512, out_bins: int = 283, use_conv_smoothing: bool = True):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_bins),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_bins),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.use_conv = use_conv_smoothing
        if use_conv_smoothing:
            self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
            with torch.no_grad():
                # simple smoothing kernel
                k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
                k = (k / k.sum()).view(1, 1, -1)
                self.conv.weight.copy_(k)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_head(z)
        logvar = self.logvar_head(z)
        if self.use_conv:
            logvar = self.conv(logvar.unsqueeze(1)).squeeze(1)
        sigma = torch.nn.functional.softplus(logvar) + 1e-6
        mu = mu / torch.clamp(self.temperature, min=1e-3)
        return mu, sigma
