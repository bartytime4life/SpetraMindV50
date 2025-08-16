# src/spectramind/models/factory.py
# -----------------------------------------------------------------------------
# Hydra-aware factory for SpectraMind V50 model components.
# Builds encoders/decoders/heads from cfg.model.* groups, logs config hashes,
# and returns a structured dict ready for training/inference pipelines.
# -----------------------------------------------------------------------------

from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

# Local imports of concrete classes
from .fgs1_mamba import FGS1MambaEncoder
from .airs_gnn import AIRSSpectralGNN
from .multi_scale_decoder import MultiScaleDecoder
from .moe_decoder import MoEDecoder
from .flow_uncertainty_head import FlowUncertaintyHead
from .spectral_corel import SpectralCOREL

logger = logging.getLogger(__name__)


@dataclass
class BuiltModels:
    """Container for all assembled model components."""
    fgs1_encoder: Any
    airs_encoder: Any
    mu_decoder: Any
    sigma_head: Any
    corel: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fgs1_encoder": self.fgs1_encoder,
            "airs_encoder": self.airs_encoder,
            "mu_decoder": self.mu_decoder,
            "sigma_head": self.sigma_head,
            "corel": self.corel,
        }


def _hash_cfg_section(cfg_section: DictConfig) -> str:
    """Create a stable hash for a config section for reproducibility logs."""
    # Convert to a JSON-serializable primitive dict
    primitive = OmegaConf.to_container(cfg_section, resolve=True)
    blob = json.dumps(primitive, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _build_fgs1_encoder(cfg: DictConfig):
    name = cfg.encoder_fgs1._name
    if name != "fgs1_mamba":
        raise ValueError(f"Unsupported FGS1 encoder: {name}")
    enc = FGS1MambaEncoder(
        input_dim=cfg.encoder_fgs1.input_dim,
        hidden_dim=cfg.encoder_fgs1.hidden_dim,
        depth=cfg.encoder_fgs1.depth,
        dropout=cfg.encoder_fgs1.dropout,
    )
    logger.info("FGS1 encoder built (%s) cfg_hash=%s", name, _hash_cfg_section(cfg.encoder_fgs1))
    return enc


def _build_airs_encoder(cfg: DictConfig):
    name = cfg.encoder_airs._name
    if name != "airs_gnn":
        raise ValueError(f"Unsupported AIRS encoder: {name}")
    enc = AIRSSpectralGNN(
        input_dim=cfg.encoder_airs.input_dim,
        hidden_dim=cfg.encoder_airs.hidden_dim,
        num_layers=cfg.encoder_airs.num_layers,
        dropout=cfg.encoder_airs.dropout,
    )
    logger.info("AIRS encoder built (%s) cfg_hash=%s", name, _hash_cfg_section(cfg.encoder_airs))
    return enc


def _build_mu_decoder(cfg: DictConfig, hidden_dim_fgs1: int, hidden_dim_airs: int):
    name = cfg.decoder_mu._name
    if name == "multi_scale_decoder":
        dec = MultiScaleDecoder(
            hidden_dim=cfg.decoder_mu.hidden_dim,
            output_bins=cfg.decoder_mu.output_bins,
        )
    elif name == "moe_decoder":
        dec = MoEDecoder(
            hidden_dim=cfg.decoder_mu.hidden_dim,
            output_bins=cfg.decoder_mu.output_bins,
            num_experts=cfg.decoder_mu.num_experts,
        )
    else:
        raise ValueError(f"Unsupported μ decoder: {name}")
    logger.info("μ decoder built (%s) cfg_hash=%s", name, _hash_cfg_section(cfg.decoder_mu))
    return dec


def _build_sigma_head(cfg: DictConfig):
    name = cfg.decoder_sigma._name
    if name != "flow_uncertainty_head":
        raise ValueError(f"Unsupported σ head: {name}")
    head = FlowUncertaintyHead(
        hidden_dim=cfg.decoder_sigma.hidden_dim,
        output_bins=cfg.decoder_sigma.output_bins,
    )
    logger.info("σ head built (%s) cfg_hash=%s", name, _hash_cfg_section(cfg.decoder_sigma))
    return head


def _build_corel(cfg: DictConfig):
    name = cfg.corel._name
    if name != "spectral_corel":
        raise ValueError(f"Unsupported COREL GNN: {name}")
    corel = SpectralCOREL(
        input_dim=cfg.corel.input_dim,
        hidden_dim=cfg.corel.hidden_dim,
        output_dim=cfg.corel.output_dim,
    )
    logger.info("COREL built (%s) cfg_hash=%s", name, _hash_cfg_section(cfg.corel))
    return corel


def build_from_cfg(cfg: DictConfig) -> BuiltModels:
    """
    Build all model components from Hydra config section `cfg.model`.

    Expected structure (see configs/model/model.yaml):
        model:
          defaults:
            - encoder_fgs1: fgs1_mamba
            - encoder_airs: airs_gnn
            - decoder_mu: multi_scale_decoder
            - decoder_sigma: flow_uncertainty_head
            - corel: spectral_corel
          ...

    Returns:
        BuiltModels: container with all ready-to-use modules.
    """
    if "model" in cfg:
        model_cfg = cfg.model
    else:
        model_cfg = cfg

    # Build encoders
    fgs1_encoder = _build_fgs1_encoder(model_cfg)
    airs_encoder = _build_airs_encoder(model_cfg)

    # Infer hidden dims to pass to decoders if needed
    hidden_dim_fgs1 = model_cfg.encoder_fgs1.hidden_dim
    hidden_dim_airs = model_cfg.encoder_airs.hidden_dim

    # Build decoders/heads
    mu_decoder = _build_mu_decoder(model_cfg, hidden_dim_fgs1, hidden_dim_airs)
    sigma_head = _build_sigma_head(model_cfg)

    # Optional COREL (enabled by default in our model group)
    corel = _build_corel(model_cfg) if "_name" in model_cfg.corel and model_cfg.corel._name else None

    built = BuiltModels(
        fgs1_encoder=fgs1_encoder,
        airs_encoder=airs_encoder,
        mu_decoder=mu_decoder,
        sigma_head=sigma_head,
        corel=corel,
    )

    # Log a single combined hash for the whole model subtree for reproducibility
    model_hash = _hash_cfg_section(model_cfg)
    logger.info("Built full SpectraMind model stack; model_cfg_hash=%s", model_hash)
    return built
