# SymbolicLoss: wrap SymbolicLogicEngine into a differentiable loss with config.
# Includes common astrophysical constraints: non-negativity, smoothness, spectral range,
# optional FFT smoothness (low-pass), and asymmetry control.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.fft as tfft

from .symbolic_logic_engine import SymbolicLogicEngine, SymbolicRule


@dataclass
class SymbolicLossConfig:
    bins: int = 283
    w_nonneg: float = 0.5
    w_smooth: float = 0.5
    w_asym: float = 0.0
    w_range: float = 0.0
    w_fft: float = 0.0
    range_lo: float = -1e6
    range_hi: float = 1e6
    fft_keep_ratio: float = 0.15  # keep low-freq power
    mask: Optional[torch.Tensor] = None  # [1 or B, bins]


def symbolic_loss_from_yaml(cfg: Dict[str, Any]) -> SymbolicLossConfig:
    return SymbolicLossConfig(
        bins=cfg.get("bins", 283),
        w_nonneg=cfg.get("w_nonneg", 0.5),
        w_smooth=cfg.get("w_smooth", 0.5),
        w_asym=cfg.get("w_asym", 0.0),
        w_range=cfg.get("w_range", 0.0),
        w_fft=cfg.get("w_fft", 0.0),
        range_lo=cfg.get("range_lo", -1e6),
        range_hi=cfg.get("range_hi", 1e6),
        fft_keep_ratio=cfg.get("fft_keep_ratio", 0.15),
        mask=None,
    )


class SymbolicLoss(nn.Module):
    def __init__(self, config: SymbolicLossConfig):
        super().__init__()
        self.cfg = config
        self.engine = SymbolicLogicEngine(bins=config.bins)

    def _fft_lowpass_violation(self, mu: torch.Tensor, keep: float) -> torch.Tensor:
        # penalize high-frequency energy beyond keep ratio of spectrum
        B, K = mu.shape
        spec = tfft.rfft(mu, dim=-1)  # [B, K//2+1]
        mag = spec.abs()
        cutoff = max(1, int(mag.shape[-1] * keep))
        hi = mag[:, cutoff:]
        # violation map back in μ-space via simple proxy: inverse rfft of zeroed low bins
        pad = spec.clone()
        pad[:, :cutoff] = 0
        recon = tfft.irfft(pad, n=K)
        v = recon.abs()
        return v

        # Alternative simple scalar: (hi**2).mean()

    def forward(self, mu: torch.Tensor) -> Dict[str, Any]:
        B, K = mu.shape
        device = mu.device
        mask = self.cfg.mask if self.cfg.mask is not None else torch.ones(1, K, device=device)

        rules: List[SymbolicRule] = []
        if self.cfg.w_nonneg > 0:
            rules.append(SymbolicRule("nonneg", mask, "nonneg", self.cfg.w_nonneg))
        if self.cfg.w_smooth > 0:
            rules.append(SymbolicRule("smooth", mask, "smooth", self.cfg.w_smooth))
        if self.cfg.w_asym > 0:
            rules.append(SymbolicRule("asym", mask, "asym", self.cfg.w_asym))
        if self.cfg.w_range > 0:
            rules.append(SymbolicRule("range", mask, "range", self.cfg.w_range, params={"lo": self.cfg.range_lo, "hi": self.cfg.range_hi}))
        out = self.engine.evaluate(mu, rules, soft=True, return_traces=True)

        if self.cfg.w_fft > 0:
            v_fft = self._fft_lowpass_violation(mu, self.cfg.fft_keep_ratio)
            fft_loss = (v_fft ** 2).mean() * self.cfg.w_fft
            out["loss"] = out["loss"] + fft_loss
            if out["traces"] is not None:
                out["traces"]["fft_violation"] = v_fft.detach()

        return out
