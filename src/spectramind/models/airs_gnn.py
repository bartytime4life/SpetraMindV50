# AIRS spectral encoder: edge-feature-aware GNN with selectable conv backends
# (GATConv/GraphConv/GCNConv/NNConv) + optional positional/region embeddings.

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATConv, GraphConv, GCNConv, NNConv, global_mean_pool
    from torch_geometric.data import Data
except Exception as e:
    GATConv = GraphConv = GCNConv = NNConv = None
    global_mean_pool = None


class _EdgeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AIRSGNN(nn.Module):
    """
    Parameters
    ----------
    in_channels : int
        Node feature dimension per wavelength bin (or per node).
    hidden_dim : int
        Hidden representation dimension.
    num_layers : int
        Number of message passing layers.
    backend : str
        One of {'GAT','Graph','GCN','NN'}.
    use_posenc : bool
        If True, concatenate sinusoidal positional encoding on input.
    use_region_emb : bool
        If True, expect region_ids tensor and add region embeddings.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        backend: str = "GAT",
        heads: int = 4,
        use_posenc: bool = True,
        max_len: int = 512,
        use_region_emb: bool = True,
        num_regions: int = 8,
        edge_attr_dim: int = 0,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if GATConv is None:
            raise ImportError(
                "torch_geometric is required for AIRSGNN. Install torch-geometric and its deps."
            )
        self.use_posenc = use_posenc
        self.use_region_emb = use_region_emb
        self.max_len = max_len
        self.region_emb = nn.Embedding(num_regions, in_channels) if use_region_emb else None

        # input projection
        self.input_proj = nn.Linear(in_channels + (in_channels if use_region_emb else 0) + (in_channels if use_posenc else 0), hidden_dim)

        # choose conv layer
        backend = backend.upper()
        self.backend = backend
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        for i in range(num_layers):
            in_dim = hidden_dim
            if backend == "GAT":
                conv = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=True)
            elif backend == "GRAPH":
                conv = GraphConv(in_dim, hidden_dim)
            elif backend == "GCN":
                conv = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
            elif backend == "NN":
                if edge_attr_dim <= 0:
                    raise ValueError("NNConv requires edge_attr_dim > 0")
                edge_mlp = _EdgeMLP(edge_attr_dim, in_dim * hidden_dim)
                conv = NNConv(in_dim, hidden_dim, edge_mlp, aggr="mean")
            else:
                raise ValueError(f"Unknown backend: {backend}")
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @staticmethod
    def _sinusoidal_posenc(n: int, d_model: int, device) -> torch.Tensor:
        pos = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
        i = torch.arange(d_model, dtype=torch.float32, device=device).unsqueeze(0)
        angle_rates = 1.0 / (10000 ** (2 * (i // 2) / d_model))
        angles = pos * angle_rates
        pe = torch.empty((n, d_model), device=device)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe

    def forward(
        self,
        x: torch.Tensor,            # [N, F]
        edge_index: torch.Tensor,   # [2, E]
        batch: Optional[torch.Tensor] = None,  # [N]
        edge_attr: Optional[torch.Tensor] = None,  # [E, D_e] for NNConv backend
        region_ids: Optional[torch.Tensor] = None, # [N] ints if use_region_emb
    ) -> torch.Tensor:
        N, F = x.shape
        device = x.device

        features = [x]
        if self.use_region_emb and region_ids is not None:
            features.append(self.region_emb(region_ids))
        if self.use_posenc:
            pe = self._sinusoidal_posenc(N, F, device)
            features.append(pe)

        x = torch.cat(features, dim=-1)
        x = self.input_proj(x)

        for conv, norm in zip(self.layers, self.norms):
            if self.backend == "NN":
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = norm(torch.relu(x))
            x = self.drop(x)

        if batch is None:
            # single-graph case: mean across nodes
            pooled = x.mean(dim=0, keepdim=True)
        else:
            pooled = global_mean_pool(x, batch)

        return self.readout(pooled)  # [B, out_dim]
