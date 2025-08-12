from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn

from .fgs1_mamba import FGS1Mamba
from .airs_gat import AIRSGAT
from .fusion import FusionConcatMLP
from .mu_decoder import MuDecoderMultiScale
from .sigma_head import SigmaHeadFlow


class SpectraMindModel(nn.Module):
    """
    Top-level V50 model wrapper that composes:
      - FGS1 encoder
      - AIRS encoder
      - Fusion
      - mu decoder
      - sigma head
    Forward signature:
      out = model(batch)
      expects dict with:
        batch["fgs1"]: (B, T, F_fgs1)
        batch["airs"]: (B, N, F_airs)  # N=283 spectral nodes typical
        batch["edges"]: optional (B, N, N, E) if GAT uses edge features
    Returns dict: {"mu": (B, N), "sigma": (B, N)}
    """
    def __init__(self, enc_fgs1, enc_airs, fusion, mu_dec, sigma_head):
        super().__init__()
        self.enc_fgs1 = enc_fgs1
        self.enc_airs = enc_airs
        self.fusion = fusion
        self.mu_dec = mu_dec
        self.sigma_head = sigma_head

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        f_fgs1 = self.enc_fgs1(batch["fgs1"])     # (B, D)
        f_airs = self.enc_airs(batch["airs"], batch.get("edges"))  # (B, N, D)
        fused = self.fusion(f_fgs1, f_airs)       # (B, N, D_fuse)
        mu = self.mu_dec(fused)                   # (B, N)
        sigma = self.sigma_head(fused)            # (B, N)
        return {"mu": mu, "sigma": sigma}


def build_model(cfg) -> nn.Module:
    """
    Build model from Hydra cfg composed from configs/model/*

    Expected cfg.model structure (after defaults resolution):
      cfg.model.latent_dim
      cfg.model.fgs1_mamba.* (type, n_layers, etc)
      cfg.model.airs_gat.*   (type, n_heads, n_layers, etc)
      cfg.model.fusion.*     (type, proj_dim, ...)
      cfg.model.mu_decoder.* (type, scales, ...)
      cfg.model.sigma_head.* (type, params...)
    """
    md = cfg.get("model") if hasattr(cfg, "get") else cfg["model"]
    D = int(md.get("latent_dim", 256))

    # Encoders
    enc_fgs1 = FGS1Mamba(
        input_dim=int(md.get("fgs1_mamba", {}).get("input_dim", 4)),
        latent_dim=D,
        n_layers=int(md.get("fgs1_mamba", {}).get("n_layers", 6)),
        bidirectional=bool(md.get("fgs1_mamba", {}).get("bidirectional", True)),
        dropout=float(md.get("fgs1_mamba", {}).get("dropout", 0.1)),
        residual=bool(md.get("fgs1_mamba", {}).get("residual", True)),
    )

    enc_airs = AIRSGAT(
        input_dim=int(md.get("airs_gat", {}).get("input_dim", D)),
        latent_dim=D,
        n_heads=int(md.get("airs_gat", {}).get("n_heads", 4)),
        n_layers=int(md.get("airs_gat", {}).get("n_layers", 3)),
        dropout=float(md.get("airs_gat", {}).get("dropout", 0.1)),
        use_edges=True,
        edge_dim=int(md.get("airs_gat", {}).get("edge_dim", 4)),
    )

    # Fusion
    fusion = FusionConcatMLP(
        fgs1_dim=D,
        airs_dim=D,
        proj_dim=int(md.get("fusion", {}).get("proj_dim", 256)),
        dropout=float(md.get("fusion", {}).get("dropout", 0.1)),
    )

    # Heads
    mu_dec = MuDecoderMultiScale(
        in_dim=int(md.get("fusion", {}).get("proj_dim", 256)),
        latent_dim=D,
        scales=list(md.get("mu_decoder", {}).get("scales", ["coarse", "mid", "fine"])),
        dropout=float(md.get("mu_decoder", {}).get("dropout", 0.1)),
    )

    sigma_head = SigmaHeadFlow(
        in_dim=int(md.get("fusion", {}).get("proj_dim", 256)),
        latent_dim=D,
        dropout=float(md.get("sigma_head", {}).get("dropout", 0.05)),
        sigma_min=float(md.get("sigma_head", {}).get("sigma_min", 1.0e-4)),
    )

    return SpectraMindModel(enc_fgs1, enc_airs, fusion, mu_dec, sigma_head)