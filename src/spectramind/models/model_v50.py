from __future__ import annotations

import torch
from torch import nn

from .decoders import MuDecoder, SigmaDecoder
from .encoders import SimpleAIRSEncoder, SimpleFGS1Encoder


class SpectraMindV50(nn.Module):
    def __init__(self, d_model: int = 128, out_bins: int = 283) -> None:
        super().__init__()
        self.fgs1 = SimpleFGS1Encoder(d_model=d_model)
        self.airs = SimpleAIRSEncoder(d_model=d_model)
        self.mu = MuDecoder(d_model=d_model, out_bins=out_bins)
        self.sigma = SigmaDecoder(d_model=d_model, out_bins=out_bins)

    def forward(
        self, fgs1: torch.Tensor, airs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_fgs1 = self.fgs1(fgs1)
        z_airs = self.airs(airs)
        mu = self.mu(z_fgs1, z_airs)
        sigma = self.sigma(z_fgs1, z_airs)
        return mu, sigma
