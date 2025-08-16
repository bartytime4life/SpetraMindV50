import torch.nn as nn


class ResidualBlock(nn.Module):
    """Simple residual block with LayerNorm."""

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.norm(self.fc(x))

