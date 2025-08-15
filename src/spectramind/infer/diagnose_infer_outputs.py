# -*- coding: utf-8 -*-
"""SpectraMind V50 — Post-Inference Diagnostics"""
from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Optional

import numpy as np
import torch

from .utils_infer import write_json


def compute_metrics(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compute core metrics; if target is None, return empty metrics."""
    if target is None:
        return {"note": "no_target", "metrics": {}}
    mu, sigma, target = mu.detach(), torch.clamp(sigma.detach(), min=1e-6), target.detach()
    gll = 0.5 * torch.log(2 * torch.pi * sigma ** 2) + (target - mu) ** 2 / (2 * sigma ** 2)
    rmse = torch.sqrt(((target - mu) ** 2).mean())
    mae = (target - mu).abs().mean()
    z = (target - mu) / sigma
    return {
        "metrics": {
            "GLL_mean": gll.mean().item(),
            "RMSE": rmse.item(),
            "MAE": mae.item(),
            "z_mean": z.mean().item(),
            "z_std": z.std(unbiased=False).item(),
        }
    }


def save_diagnostics(
    out_dir: pathlib.Path,
    metrics: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = dict(metrics)
    if extra:
        payload["extra"] = extra
    write_json(out_dir / "diagnostics_summary.json", payload)
