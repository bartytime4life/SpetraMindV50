# Attention/attribution diagnostic utilities to fuse SHAP/attention with symbolic rule traces.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


@dataclass
class AttentionTrace:
    shap: Optional[torch.Tensor] = None   # [B, bins]
    attn: Optional[torch.Tensor] = None   # [B, bins]
    symb: Optional[torch.Tensor] = None   # [B, bins]


class FusionAttentionDiagnostics(nn.Module):
    def __init__(self, mode: str = "zscore_fuse"):
        super().__init__()
        self.mode = mode

    @staticmethod
    def _z(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

    def forward(self, trace: AttentionTrace) -> Dict[str, torch.Tensor]:
        """
        Returns fused attribution suitable for plotting and thresholding.
        """
        comps = []
        if trace.shap is not None:
            comps.append(self._z(trace.shap.abs()))
        if trace.attn is not None:
            comps.append(self._z(trace.attn))
        if trace.symb is not None:
            comps.append(self._z(trace.symb))

        if not comps:
            raise ValueError("No components provided for fusion.")

        fused = torch.stack(comps, dim=0).mean(dim=0)  # [B, bins]
        top_bins = torch.topk(fused, k=max(1, fused.shape[-1] // 20), dim=-1).indices  # top 5% default
        return {"fused": fused, "top_bins": top_bins}
