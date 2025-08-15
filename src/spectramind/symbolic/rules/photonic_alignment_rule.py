# SPDX-License-Identifier: Apache-2.0

"""Cross-instrument alignment rule between spectral and photometric data."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base_rule import SymbolicRule
from .utils import safe_corrcoef


class PhotonicAlignmentRule(SymbolicRule):
    """Enforce alignment between smoothed spectra and FGS1 transit curves."""

    def __init__(
        self,
        weight: float = 1.0,
        smoothing_window: int = 11,
        continuum_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(name="photonic_alignment", weight=weight, **kwargs)
        self.smoothing_window = int(smoothing_window)
        self.continuum_weight = float(continuum_weight)

    def _moving_avg(self, x: Tensor, k: int) -> Tensor:
        if k <= 1:
            return x
        kernel = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / float(k)
        x_exp = x.unsqueeze(1)
        pad = (k // 2, k - 1 - (k // 2))
        y = F.pad(x_exp, (pad[0], pad[1]), mode="replicate")
        return F.conv1d(y, kernel).squeeze(1)

    def evaluate_map(
        self,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> Tensor:
        assert (
            isinstance(metadata, dict) and "fgs1_curve" in metadata
        ), "PhotonicAlignmentRule requires metadata['fgs1_curve']"

        fgs1 = metadata["fgs1_curve"]
        if not torch.is_tensor(fgs1):
            fgs1 = torch.as_tensor(fgs1, dtype=mu.dtype, device=mu.device)
        if fgs1.dim() == 1:
            fgs1 = fgs1.unsqueeze(0)
        assert fgs1.size(0) == mu.size(0), "Batch size mismatch"

        mu_smooth = self._moving_avg(
            mu, metadata.get("airs_smoothing", self.smoothing_window)
        )
        ms = (mu_smooth - mu_smooth.mean(dim=1, keepdim=True)) / (
            mu_smooth.std(dim=1, keepdim=True).clamp_min(1e-6)
        )
        ft = (fgs1 - fgs1.mean(dim=1, keepdim=True)) / (
            fgs1.std(dim=1, keepdim=True).clamp_min(1e-6)
        )

        ms_low = ms.mean(dim=1)
        ft_low = ft.mean(dim=1)
        corr = safe_corrcoef(ms_low, ft_low)

        violation = (1.0 - corr).clamp_min(0.0).unsqueeze(1).expand_as(mu)

        mask = metadata.get("continuum_only_mask")
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, dtype=mu.dtype, device=mu.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand_as(mu)
            violation = violation * mask * self.continuum_weight

        return violation.clamp_min(0.0)


__all__ = ["PhotonicAlignmentRule"]
