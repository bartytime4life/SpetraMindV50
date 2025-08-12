from __future__ import annotations
import torch
import torch.nn as nn


class FusionConcatMLP(nn.Module):
    """
    Concatenate global FGS1 vector to each AIRS node embedding, then project.
      fgs1: (B, D1)
      airs: (B, N, D2)
    Output: (B, N, P)
    """
    def __init__(self, fgs1_dim: int, airs_dim: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fgs1_dim + airs_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, fgs1: torch.Tensor, airs: torch.Tensor) -> torch.Tensor:
        B, N, D2 = airs.shape
        fgs1_expand = fgs1.unsqueeze(1).expand(-1, N, -1)  # (B, N, D1)
        fused = torch.cat([fgs1_expand, airs], dim=-1)
        return self.mlp(fused)