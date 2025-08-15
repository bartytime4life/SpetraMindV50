from __future__ import annotations

import torch
from torch import nn


class MuDecoder(nn.Module):
    def __init__(self, d_model: int = 128, out_bins: int = 283) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_bins),
        )

    def forward(self, z_fgs1: torch.Tensor, z_airs: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_fgs1, z_airs], dim=-1)
        return self.net(z)


class SigmaDecoder(nn.Module):
    def __init__(
        self, d_model: int = 128, out_bins: int = 283, min_sigma: float = 1e-3
    ) -> None:
        super().__init__()
        self.min_sigma = min_sigma
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_bins),
        )

    def forward(self, z_fgs1: torch.Tensor, z_airs: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_fgs1, z_airs], dim=-1)
        return torch.nn.functional.softplus(self.net(z)) + self.min_sigma
