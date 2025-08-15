# Fusion ensemble that combines rule-based predictor outputs with the NN predictor logits
# to produce calibrated violation probabilities and a ranked list per planet.

from __future__ import annotations
from typing import Dict, Any

import torch
import torch.nn as nn


class SymbolicFusionPredictor(nn.Module):
    def __init__(self, num_rules: int, method: str = "logit_avg", temperature: float = 1.0):
        super().__init__()
        self.num_rules = num_rules
        self.method = method
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, rb_scores: torch.Tensor, nn_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        rb_scores: [B, R] (rule-based numeric scores, larger=worse)
        nn_logits: [B, R]
        Returns:
            probs: [B, R] in [0,1]
            ranks: [B, R] argsort descending by probs
        """
        # normalize rule-based scores to logits via affine map
        rb_norm = (rb_scores - rb_scores.mean(dim=0, keepdim=True)) / (rb_scores.std(dim=0, keepdim=True) + 1e-6)
        rb_logits = rb_norm

        if self.method == "logit_avg":
            logits = (rb_logits + nn_logits) / 2.0
        elif self.method == "stack":
            logits = rb_logits + nn_logits
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

        logits = logits / torch.clamp(self.temperature, min=1e-3)
        probs = torch.sigmoid(logits)
        ranks = torch.argsort(probs, dim=-1, descending=True)
        return {"probs": probs, "ranks": ranks}
