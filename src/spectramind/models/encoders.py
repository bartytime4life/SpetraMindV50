from __future__ import annotations

import torch
from torch import nn


class SimpleFGS1Encoder(nn.Module):
    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, d_model),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, H, W) -> (B,1,H,W)
        if x.dim() == 5:
            x = x.mean(1, keepdim=False).unsqueeze(1)
        return self.net(x)


class SimpleAIRSEncoder(nn.Module):
    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(128 * 16, d_model),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 32, 356) where 32 is treated as channels
        return self.net(x)
