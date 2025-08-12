from __future__ import annotations

class AIRSSpectralGNN:
    """Placeholder spectral GNN; real version will use torch_geometric GAT/NNConv."""

    def __init__(self, in_dim: int = 64, latent_dim: int = 128, use_gat: bool = True) -> None:
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.use_gat = use_gat

    def __repr__(self) -> str:
        return f"AIRSSpectralGNN(in_dim={self.in_dim}, latent_dim={self.latent_dim}, use_gat={self.use_gat})"

    def encode(self, num_nodes: int = 283) -> list[float]:
        return [0.0 for _ in range(self.latent_dim)]
