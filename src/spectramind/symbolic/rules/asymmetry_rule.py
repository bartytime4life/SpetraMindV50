# SPDX-License-Identifier: Apache-2.0

"""Penalize excessive line-shape asymmetry around known band centers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from .base_rule import SymbolicRule
from .utils import build_centers_mask, get_wavelengths


class AsymmetryRule(SymbolicRule):
    """Compare left vs right slopes and areas around spectral centers."""

    def __init__(
        self,
        weight: float = 1.0,
        centers_um: Optional[List[float]] = None,
        window_um: float = 0.08,
        slope_weight: float = 1.0,
        area_weight: float = 0.5,
        min_bins: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(name="asymmetry", weight=weight, **kwargs)
        self.centers_um = centers_um or [1.4, 1.9, 2.3, 3.3, 4.3, 6.3]
        self.window_um = float(window_um)
        self.slope_weight = float(slope_weight)
        self.area_weight = float(area_weight)
        self.min_bins = int(min_bins)

    def evaluate_map(
        self,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> Tensor:
        assert mu.dim() == 2, "mu must be (B, N)"
        B, N = mu.shape
        wl = get_wavelengths(metadata, N, device=mu.device)

        violation = torch.zeros_like(mu)

        for c in self.centers_um:
            mask, left_idx, right_idx = build_centers_mask(wl, c, self.window_um)
            if mask is None:
                continue
            valid = (right_idx - left_idx + 1) >= (2 * self.min_bins + 1)
            if valid.sum() == 0:
                continue
            for b in range(B):
                if not bool(valid[b]):
                    continue
                li = int(left_idx[b].item())
                ri = int(right_idx[b].item())
                mid = (li + ri) // 2
                left = mu[b, li:mid]
                right = mu[b, mid + 1 : ri + 1]
                if left.numel() < self.min_bins or right.numel() < self.min_bins:
                    continue
                slope_left = (left[-1] - left[0]) / max(1, left.numel() - 1)
                slope_right = (right[-1] - right[0]) / max(1, right.numel() - 1)
                slope_asym = (slope_left - slope_right).abs()
                area_asym = (left.mean() - right.mean()).abs()
                local_violation = (
                    self.slope_weight * slope_asym + self.area_weight * area_asym
                )
                violation[b, li : ri + 1] += local_violation

        return violation.clamp_min(0.0)


__all__ = ["AsymmetryRule"]
