# SymbolicLogicEngine: vectorized rule evaluation on μ spectra with masks, weights,
# soft/hard modes, and trace outputs suitable for diagnostics dashboards.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SymbolicRule:
    name: str
    mask: torch.Tensor           # [B, bins] boolean or float in {0,1}
    kind: str                    # 'nonneg', 'smooth', 'asym', 'range', 'custom'
    weight: float = 1.0
    params: Optional[Dict[str, Any]] = None


class SymbolicLogicEngine(nn.Module):
    def __init__(self, bins: int = 283, device: Optional[torch.device] = None):
        super().__init__()
        self.bins = bins
        self.device = device
        self.register_buffer("_dummy", torch.zeros(1))  # ensures device tracking

    def _ensure(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self._dummy.device)

    def evaluate(
        self,
        mu: torch.Tensor,                     # [B, bins]
        rules: List[SymbolicRule],
        soft: bool = True,
        return_traces: bool = True,
    ) -> Dict[str, Any]:
        mu = self._ensure(mu)
        traces: Dict[str, torch.Tensor] = {}
        per_rule_loss = []

        for r in rules:
            m = self._ensure(r.mask).float()  # broadcastable to mu
            w = float(r.weight)
            if r.kind == "nonneg":
                v = torch.clamp(-mu, min=0.0) * m
                l = (v ** 2).mean()
            elif r.kind == "smooth":
                # L2 of first difference in masked region
                d = (mu[:, 1:] - mu[:, :-1])
                m2 = (m[:, 1:] * m[:, :-1])
                l = ((d * m2) ** 2).mean()
                v = torch.zeros_like(mu)
                v[:, 1:] = torch.abs(d) * m2
            elif r.kind == "asym":
                # encourage left-right similarity unless mask suggests otherwise
                rev = torch.flip(mu, dims=[-1])
                v = (mu - rev) * m
                l = (v ** 2).mean()
            elif r.kind == "range":
                lo = float(r.params.get("lo", -1e9))
                hi = float(r.params.get("hi", +1e9))
                v = (torch.clamp(lo - mu, min=0) + torch.clamp(mu - hi, min=0)) * m
                l = (v ** 2).mean()
            elif r.kind == "custom":
                # expects callable in params['fn'](mu, mask)->violation map
                fn = r.params["fn"]
                v = fn(mu, m)
                l = (v ** 2).mean()
            else:
                raise ValueError(f"Unknown rule kind: {r.kind}")

            if return_traces:
                traces[r.name] = v.detach()
            per_rule_loss.append(w * l)

        total = torch.stack(per_rule_loss).sum() if per_rule_loss else mu.new_tensor(0.0)
        return {
            "loss": total,
            "per_rule": torch.stack(per_rule_loss) if per_rule_loss else mu.new_zeros(1),
            "traces": traces if return_traces else None,
        }
