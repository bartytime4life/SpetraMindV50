# SPDX-License-Identifier: Apache-2.0

"""Frequency-domain constraint using FFT power spectrum."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from .base_rule import SymbolicRule
from .utils import rfft_power_spectrum


class FFTSpectralRule(SymbolicRule):
    """Operate in frequency domain to limit high-frequency power."""

    def __init__(
        self,
        weight: float = 1.0,
        tail_frac: float = 0.3,
        power_anomaly_weight: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(name="fft_spectral", weight=weight, **kwargs)
        self.tail_frac = float(tail_frac)
        self.power_anomaly_weight = float(power_anomaly_weight)

    def evaluate_map(
        self,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> Tensor:
        assert mu.dim() == 2, "mu must be (B, N)"
        B, N = mu.shape
        P = rfft_power_spectrum(mu)
        F = P.size(1)
        tail_len = max(1, int(self.tail_frac * F))
        tail_power = P[:, -tail_len:].sum(dim=1)
        total_power = P.sum(dim=1).clamp_min(1e-12)
        ratio = (tail_power / total_power).unsqueeze(1)

        violation = ratio.expand(B, N)

        neigh = torch.nn.functional.pad(P, (1, 1), mode="replicate")
        local_std = (neigh[:, 2:] - neigh[:, :-2]).abs().mean(dim=1, keepdim=True)
        anomaly = self.power_anomaly_weight * local_std
        violation = violation + anomaly.expand_as(violation)

        return violation.clamp_min(0.0)


__all__ = ["FFTSpectralRule"]
