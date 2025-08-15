# FGS1 encoder: prefer Mamba-SSM; gracefully fallback to TransformerEncoder if unavailable.

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class _TransformerFallback(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=4*d_model, dropout=dropout, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        z = self.proj(x)
        z = self.encoder(z)
        z = self.norm(z.mean(dim=1))
        return z


class FGS1MambaEncoder(nn.Module):
    """
    Inputs
    ------
    x : [B, T, C]   (Batches of temporal sequences with C channels)
    mask : Optional boolean mask [B, T] (True = keep)

    Returns
    -------
    [B, d_model] pooled representation.
    """
    def __init__(
        self,
        in_dim: int = 32,
        d_model: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        fallback_nhead: int = 8,
    ):
        super().__init__()
        self.use_mamba = False
        try:
            from mamba_ssm import Mamba
            self.use_mamba = True
        except Exception:
            self.use_mamba = False

        if self.use_mamba:
            self.input = nn.Linear(in_dim, d_model)
            self.blocks = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),  # lightweight pre-proj
                nn.GELU(),
                # Mamba block
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
            ) for _ in range(num_layers)])
            self.head_norm = nn.LayerNorm(d_model)
        else:
            self.fallback = _TransformerFallback(in_dim, d_model, fallback_nhead, num_layers, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_mamba:
            return self.fallback(x)

        # Mamba path
        z = self.input(x)  # [B, T, d_model]
        for blk in self.blocks:
            z = z + blk(z)
        if mask is not None:
            # mask True=keep, False=drop -> create weights
            w = mask.float().unsqueeze(-1)  # [B,T,1]
            denom = w.sum(dim=1).clamp(min=1.0)
            pooled = (z * w).sum(dim=1) / denom
        else:
            pooled = z.mean(dim=1)
        return self.head_norm(pooled)
