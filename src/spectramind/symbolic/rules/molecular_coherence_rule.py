# SPDX-License-Identifier: Apache-2.0

"""Encourage molecular-feature consistency across known absorption bands."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from .base_rule import SymbolicRule
from .utils import build_band_masks, get_wavelengths, robust_baseline

_DEFAULT_BANDS: Dict[str, List[float]] = {
    "H2O": [1.4, 1.9, 2.7, 6.3],
    "CO2": [2.0, 4.3],
    "CH4": [1.7, 2.3, 3.3, 7.7],
    "CO": [2.3, 4.7],
    "NH3": [1.5, 2.0, 10.5],
    "HCN": [3.0],
    "TiO": [0.7],
    "VO": [0.8],
}


class MolecularCoherenceRule(SymbolicRule):
    """Uplift and coherence constraints across molecular bands."""

    def __init__(
        self,
        weight: float = 1.0,
        molecules: Optional[Dict[str, List[float]]] = None,
        band_width: float = 0.05,
        margin: float = 0.0,
        coherence_weight: float = 0.5,
        use_sigma_guard: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name="molecular_coherence", weight=weight, **kwargs)
        self.molecules = molecules or _DEFAULT_BANDS
        self.band_width = float(band_width)
        self.margin = float(margin)
        self.coherence_weight = float(coherence_weight)
        self.use_sigma_guard = bool(use_sigma_guard)

    def evaluate_map(
        self,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> Tensor:
        assert mu.dim() == 2, "mu must be (B, N)"
        B, N = mu.shape
        wl = get_wavelengths(metadata, N, device=mu.device)

        band_masks = build_band_masks(
            wl, self.molecules, band_half_width=self.band_width
        )

        combined = torch.zeros_like(mu)
        for m in band_masks.values():
            combined = torch.maximum(combined, m)
        out_of_band_mask = (combined < 0.5).float()
        baseline = robust_baseline(mu, out_of_band_mask)

        if sigma is not None and self.use_sigma_guard:
            s_med = sigma.median(dim=1, keepdim=True).values.clamp_min(1e-6)
            sigma_guard = (s_med / sigma.clamp_min(1e-6)).clamp(0.0, 1.5).clamp_max(1.0)
        else:
            sigma_guard = torch.ones_like(mu)

        violation = torch.zeros_like(mu)

        for mask in band_masks.values():
            if mask.sum() == 0:
                continue
            uplift = (baseline + self.margin) - mu
            uplift = torch.relu(uplift) * mask * sigma_guard
            band_mean = (mu * mask).sum(dim=1, keepdim=True) / (
                mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            coh = torch.relu((baseline + self.margin) - band_mean)
            coherence_penalty = self.coherence_weight * coh * mask
            violation = violation + uplift + coherence_penalty

        return violation.clamp_min(0.0)


__all__ = ["MolecularCoherenceRule"]
