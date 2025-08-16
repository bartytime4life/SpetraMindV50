"""
SpectraMind V50 Model Package
=============================

This package contains the core neural architectures for the NeurIPS 2025 Ariel Data Challenge.
It integrates sequence models (Mamba SSM), spectral GNNs, multi-scale decoders, symbolic-aware
loss heads, and utility layers. All models follow Hydra-safe config loading, logging, and
reproducibility standards.
"""

from .fgs1_mamba import FGS1MambaEncoder
from .airs_gnn import AIRSSpectralGNN
from .multi_scale_decoder import MultiScaleDecoder
from .moe_decoder import MoEDecoder
from .flow_uncertainty_head import FlowUncertaintyHead
from .spectral_corel import SpectralCOREL
from .base_model import SpectraMindModel
from .model_registry import get_model_class, register_model

__all__ = [
    "FGS1MambaEncoder",
    "AIRSSpectralGNN",
    "MultiScaleDecoder",
    "MoEDecoder",
    "FlowUncertaintyHead",
    "SpectralCOREL",
    "SpectraMindModel",
    "get_model_class",
    "register_model",
]


from .factory import build_from_cfg, BuiltModels
