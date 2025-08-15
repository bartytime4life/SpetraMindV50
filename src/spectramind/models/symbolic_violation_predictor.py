# Rule-based symbolic violation scorer.
# Consumes μ spectra and the traces produced by SymbolicLoss to emit per-rule scores
# and boolean masks for dashboards, CSV/JSON export layers, and training feedback loops.

from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class SymbolicViolationPredictor(nn.Module):
    def __init__(self, threshold: float = 0.0, aggregate: str = "mean"):
        super().__init__()
        self.threshold = threshold
        self.aggregate = aggregate  # 'mean', 'max', 'sum'

    def forward(self, traces: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        scores: Dict[str, float] = {}
        masks: Dict[str, torch.Tensor] = {}

        for name, v in traces.items():
            # v: [B, bins] violation magnitude per bin
            if self.aggregate == "mean":
                s = v.mean(dim=-1)
            elif self.aggregate == "max":
                s = v.amax(dim=-1)
            elif self.aggregate == "sum":
                s = v.sum(dim=-1)
            else:
                raise ValueError(f"Unknown aggregate: {self.aggregate}")

            scores[name] = s.detach().cpu()
            masks[name] = (v > self.threshold).detach()

        return {"scores": scores, "masks": masks}
