from __future__ import annotations
import torch
import torch.nn as nn


class SimpleSSMBlock(nn.Module):
    """
    Very small SSM-ish block that is TorchScript-friendly and fast.
    Not a true Mamba; acts as a placeholder with gated mixing.
    Input: (B, T, F) -> (B, T, D)
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin_f = nn.Linear(in_dim, out_dim)
        self.lin_g = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.lin_f(x)
        g = torch.sigmoid(self.lin_g(x))
        y = f * g
        y = self.proj(self.act(y))
        return self.dropout(y)


class FGS1Mamba(nn.Module):
    """
    FGS1 encoder stub. Pools over time to produce a single (B, D) vector.
    """
    def __init__(self, input_dim: int, latent_dim: int, n_layers: int = 6,
                 bidirectional: bool = True, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(SimpleSSMBlock(in_dim, latent_dim, dropout))
            in_dim = latent_dim
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.latent_dim = latent_dim
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        y = self.net(x)                     # (B, T, D)
        y = y.transpose(1, 2)               # (B, D, T)
        y = self.pool(y).squeeze(-1)        # (B, D)
        return y