from __future__ import annotations
import torch
import torch.nn as nn


class SimpleGATLayer(nn.Module):
    """
    Lightweight graph attention over spectral nodes:
      - h: (B, N, D)
      - edges: optional (B, N, N, E) used to modulate attention logits
    """
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1, edge_dim: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        # Edge projection to a scalar bias per head
        self.edge_proj = nn.Linear(edge_dim, n_heads)

    def forward(self, h: torch.Tensor, edges: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = h.shape
        q = self.q(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = self.k(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N, N)
        if edges is not None:
            # edges: (B, N, N, E) -> bias per head
            bias = self.edge_proj(edges)  # (B, N, N, H)
            bias = bias.permute(0, 3, 1, 2)  # (B, H, N, N)
            attn_logits = attn_logits + bias

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out(out)
        return out


class AIRSGAT(nn.Module):
    """
    AIRS spectral encoder stub with stacked SimpleGATLayers.
    Produces nodewise embeddings (B, N, D).
    """
    def __init__(self, input_dim: int, latent_dim: int, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = 0.1, use_edges: bool = True, edge_dim: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, latent_dim)
        self.layers = nn.ModuleList([
            SimpleGATLayer(latent_dim, n_heads=n_heads, dropout=dropout, edge_dim=edge_dim)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)
        self.use_edges = use_edges

    def forward(self, x: torch.Tensor, edges: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, F) -> (B, N, D)
        h = self.in_proj(x)
        for layer in self.layers:
            h = h + layer(h, edges if self.use_edges else None)
        return self.norm(h)