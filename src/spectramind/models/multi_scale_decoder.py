# Multi-scale μ decoder: coarse-to-fine heads + residual merge + optional skip/fusion.

from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn


class _ScaleHead(nn.Module):
    def __init__(self, in_dim: int, out_bins: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, out_bins),
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MultiScaleDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 512,
        out_bins: int = 283,
        scales: Sequence[int] = (1, 2, 4),
        base_width: int = 512,
        dropout: float = 0.1,
        use_residual_merge: bool = True,
    ):
        super().__init__()
        self.scales = tuple(scales)
        self.use_residual_merge = use_residual_merge
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleList()
        for s in self.scales:
            width = max(base_width // s, 64)
            self.heads.append(_ScaleHead(in_dim, out_bins, width))

        if use_residual_merge:
            self.merge = nn.Parameter(torch.zeros(len(self.scales)))
            nn.init.uniform_(self.merge, 0.25, 0.75)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        preds = [self.dropout(h(z)) for h in self.heads]  # list of [B, out_bins]
        if self.use_residual_merge:
            # learnable convex combination (softmax over weights)
            w = torch.softmax(self.merge, dim=0)  # [S]
            stacked = torch.stack(preds, dim=-1)  # [B, out_bins, S]
            return (stacked * w).sum(dim=-1)
        else:
            return torch.stack(preds, dim=-1).mean(dim=-1)
