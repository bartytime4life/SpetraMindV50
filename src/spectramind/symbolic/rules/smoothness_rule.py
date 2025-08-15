# SPDX-License-Identifier: Apache-2.0

"""Penalize high-frequency roughness in spectra via second-difference and FFT tail power."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor

from .base_rule import SymbolicRule
from .utils import build_optional_region_mask, fft_rfft_power


class SmoothnessRule(SymbolicRule):
    """Encourages spectral smoothness and suppresses spurious ripples."""

    def __init__(
        self,
        weight: float = 1.0,
        laplace_weight: float = 1.0,
        fft_weight: float = 0.5,
        fft_tail_frac: float = 0.25,
        region: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(name="smoothness", weight=weight, **kwargs)
        self.laplace_weight = float(laplace_weight)
        self.fft_weight = float(fft_weight)
        self.fft_tail_frac = float(fft_tail_frac)
        self.region = region or {}

    def _second_diff(self, x: Tensor) -> Tensor:
        x_pad = F.pad(x.unsqueeze(1), (1, 1), mode="replicate").squeeze(1)
        return x_pad[:, 2:] - 2.0 * x_pad[:, 1:-1] + x_pad[:, :-2]

    def evaluate_map(
        self,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> Tensor:
        assert mu.dim() == 2, "mu must be (B, N)"
        B, N = mu.shape

        region_mask = build_optional_region_mask(metadata, N, self.region)

        d2 = self._second_diff(mu)
        laplace_violation = d2.pow(2)
        if region_mask is not None:
            laplace_violation = laplace_violation * region_mask

        tail_ratio = fft_rfft_power(mu, tail_frac=self.fft_tail_frac)
        fft_violation = tail_ratio.clamp_min(0.0).unsqueeze(1).expand(B, N)

        violation = (
            self.laplace_weight * laplace_violation + self.fft_weight * fft_violation
        )
        return violation.clamp_min(0.0)


__all__ = ["SmoothnessRule"]
