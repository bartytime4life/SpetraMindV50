# Neural symbolic violation predictor: learns to map μ spectra (and optional metadata)
# to per-rule violation logits. Trains on supervision derived from engine traces.

from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class SymbolicViolationPredictorNN(nn.Module):
    def __init__(self, bins: int = 283, num_rules: int = 8, width: int = 512, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        last = bins
        for i in range(depth):
            layers += [nn.Linear(last, width), nn.GELU(), nn.Dropout(dropout)]
            last = width
        layers += [nn.Linear(last, num_rules)]
        self.net = nn.Sequential(*layers)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        # returns logits [B, num_rules]
        return self.net(mu)

    @staticmethod
    def loss_fn(logits: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # BCE with logits for multi-label violation targets (0/1 per rule)
        return nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight, reduction="mean")
