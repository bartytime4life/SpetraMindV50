# SPDX-License-Identifier: Apache-2.0

"""Composite wrapper to combine multiple rules."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from .base_rule import SymbolicRule


class CompositeRule(SymbolicRule):
    """Combine multiple ``SymbolicRule`` instances into one."""

    def __init__(
        self,
        rules: List[SymbolicRule],
        weight: float = 1.0,
        mode: str = "sum",
        inner_weights: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(name="composite", weight=weight, **kwargs)
        assert len(rules) > 0, "CompositeRule requires at least one sub-rule"
        self.rules = torch.nn.ModuleList(rules)
        self.mode = str(mode)
        self.inner_weights = inner_weights
        if self.mode == "weighted":
            assert inner_weights is not None and len(inner_weights) == len(rules)

    def evaluate_map(
        self,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> Tensor:
        maps = [r(mu, sigma, metadata).violation_map for r in self.rules]
        ref_shape = maps[0].shape
        for m in maps:
            assert m.shape == ref_shape, "All sub-rule maps must share the same shape"
        if self.mode == "max":
            vmap = torch.stack(maps, dim=0).max(dim=0).values
        elif self.mode == "weighted":
            stacked = torch.stack(maps, dim=0)
            w = torch.as_tensor(
                self.inner_weights, dtype=stacked.dtype, device=stacked.device
            ).view(-1, 1, 1)
            vmap = (w * stacked).sum(dim=0)
        else:
            vmap = torch.stack(maps, dim=0).sum(dim=0)
        return vmap.clamp_min(0.0)

    def reduce_loss(
        self,
        violation_map: Tensor,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> Tensor:
        return self.weight * violation_map.mean()


__all__ = ["CompositeRule"]
