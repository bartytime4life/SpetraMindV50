# -*- coding: utf-8 -*-
"""SpectraMind V50 — Uncertainty Calibration"""
from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .utils_infer import append_to_debug_log, write_json, write_jsonl_event, InferenceConfig


class TemperatureScaler(torch.nn.Module):
    """Per-bin or global temperature scaling for sigma."""

    def __init__(self, bins: int, per_bin: bool = True):
        super().__init__()
        self.bins = bins
        self.per_bin = per_bin
        if per_bin:
            self.log_T = torch.nn.Parameter(torch.zeros(bins))
        else:
            self.log_T = torch.nn.Parameter(torch.zeros(1))

    def fit(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
        lr: float = 0.1,
        steps: int = 200,
    ) -> torch.Tensor:
        assert mu.shape == sigma.shape == target.shape
        opt = torch.optim.Adam([self.log_T], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            T = self.temperature().view(1, -1 if self.per_bin else 1)
            cal_sigma = torch.clamp(sigma * T, min=1e-6)
            loss = 0.5 * torch.log(2 * torch.pi * cal_sigma ** 2) + (target - mu) ** 2 / (
                2 * cal_sigma ** 2
            )
            loss = loss.mean()
            loss.backward()
            opt.step()
        return self.temperature().detach()

    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_T)

    def apply(self, sigma: torch.Tensor) -> torch.Tensor:
        T = self.temperature().view(1, -1 if self.per_bin else 1)
        return torch.clamp(sigma * T, min=1e-6)


class CorelCalibrator:
    """Thin wrapper to call external Spectral COREl GNN calibrator if present."""

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self._impl = None
        try:
            mod = __import__("spectramind.calibration.corel", fromlist=["SpectralCOREL"])
            self._impl = getattr(mod, "SpectralCOREL")
        except Exception:
            self._impl = None

    def fit_predict(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logger = logging.getLogger("spectramind.infer")
        if self._impl is None:
            logger.warning("COREL calibrator not available; sigma passthrough.")
            return sigma
        try:
            instance = self._impl(
                cfg=self.cfg, bins=mu.shape[1], **(self.cfg.calibration.get("corel", {}))
            )
            return instance.fit_predict(mu=mu, sigma=sigma, y=target)
        except Exception as e:  # pragma: no cover
            logger.exception("COREL calibrator failed; sigma passthrough: %s", e)
            return sigma

def calibrate_predictions(
    cfg: InferenceConfig,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    calib: Optional[Dict[str, torch.Tensor]],
    out_dir: pathlib.Path,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Apply configured calibration steps; return calibrated (mu, sigma) and summary."""
    logger = logging.getLogger("spectramind.infer")
    sigma = torch.clamp(sigma, min=1e-6)
    summary: Dict[str, Any] = {"steps": []}

    tconf = cfg.calibration.get("temperature", {"enabled": True, "per_bin": True})
    if tconf.get("enabled", True) and calib is not None and "target" in calib:
        scaler = TemperatureScaler(
            bins=mu.shape[1], per_bin=bool(tconf.get("per_bin", True))
        )
        T = scaler.fit(
            mu=mu,
            sigma=sigma,
            target=calib["target"],
            lr=float(tconf.get("lr", 0.1)),
            steps=int(tconf.get("steps", 200)),
        )
        sigma = scaler.apply(sigma)
        summary["steps"].append(
            {
                "kind": "temperature_scaling",
                "per_bin": scaler.per_bin,
                "T": T.detach().cpu().tolist(),
            }
        )
        logger.info("Applied temperature scaling (per_bin=%s).", scaler.per_bin)
    else:
        logger.info(
            "Temperature scaling skipped (enabled=%s, target=%s).",
            tconf.get("enabled", True),
            calib is not None and "target" in calib,
        )

    if cfg.calibration.get("corel", {}).get("enabled", False) and calib is not None and "target" in calib:
        corel = CorelCalibrator(cfg)
        sigma = corel.fit_predict(mu=mu, sigma=sigma, target=calib["target"])
        summary["steps"].append({"kind": "corel"})
        logger.info("Applied COREL calibration.")
    else:
        logger.info("COREL calibration skipped or unavailable.")

    write_json(out_dir / "calibration_summary.json", summary)
    return mu, sigma, summary
