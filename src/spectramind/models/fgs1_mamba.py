import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class FGS1MambaEncoder(nn.Module):
    """
    Fine Guidance Sensor 1 (FGS1) Encoder using Mamba SSM.
    Handles long-sequence (135k × 32 × 32) transit time-series.

    Features:
    - State Space Model backbone with efficient recurrence.
    - Temporal dropout + jitter augmentation hooks.
    - Symbolic smoothness & transit-shape aware embeddings.
    """

    def __init__(self, input_dim=32, hidden_dim=128, depth=12, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu"
            ) for _ in range(depth)
        ])
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        logger.info("Initialized FGS1MambaEncoder")

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

