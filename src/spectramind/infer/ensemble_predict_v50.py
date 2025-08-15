# -*- coding: utf-8 -*-
"""SpectraMind V50 — Ensemble Prediction Aggregator"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .utils_infer import write_json


def aggregate_ensemble(
    preds: List[Tuple[torch.Tensor, torch.Tensor]],
    method_mu: str = "mean",
    method_sigma: str = "geom_mean",
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Aggregate predictions from K models."""
    K = len(preds)
    assert K >= 1, "Need at least one prediction to aggregate."
    mus = torch.stack([p[0] for p in preds], dim=0)
    sig = torch.stack([p[1] for p in preds], dim=0)

    if weights is None:
        w = torch.ones((K, 1, 1), dtype=mus.dtype, device=mus.device) / K
    else:
        w = weights
        if w.ndim == 1:
            w = w.view(K, 1, 1)
        elif w.ndim == 2:
            w = w.view(K, 1, -1)
        w = w / (w.sum(dim=0, keepdim=True) + 1e-9)

    if method_mu == "median":
        mu_agg = mus.median(dim=0).values
    else:
        mu_agg = (w * mus).sum(dim=0)

    if method_sigma == "mean":
        sigma_agg = (w * sig).sum(dim=0)
    else:
        sigma_agg = torch.exp((w * torch.log(torch.clamp(sig, min=1e-9))).sum(dim=0))

    summary = {
        "K": K,
        "method_mu": method_mu,
        "method_sigma": method_sigma,
        "weights_shape": list(w.shape),
    }
    return mu_agg, sigma_agg, summary
