#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SpectraMind V50 — Hydra Structured Configs for configs/model/src/spectramind/*.

This module defines strict dataclass schemas for all "model/src/spectramind" component
configs (FGS1 Mamba SSM encoder, AIRS GNN encoder, μ multi-scale decoder, σ head,
symbolic loss pack, and fusion). It also registers them with Hydra's ConfigStore and
provides helpers to validate/merge user YAMLs against these schemas.

Design goals:
    • Strict typing with meaningful ValueErrors on invalid ranges/types.
    • Hydra-friendly (ConfigStore registration) but also usable standalone for validation.
    • Slot attributes for memory/perf; defaults mirror your YAMLs; safe numeric floors.
    • Ready to import from CLI or training code to ensure early, consistent validation.

Usage (programmatic):
    from spectramind.config.schemas_spectramind import (
        register_spectramind_config_schemas,
        load_and_validate_yaml,
        SpectraMindModelSrcConfig,
    )

    register_spectramind_config_schemas()
    merged_cfg = load_and_validate_yaml("configs/model/src/spectramind/fgs1_mamba.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

def _assert_positive(name: str, value: float, strict: bool = True) -> None:
    if strict and not (value > 0):
        raise ValueError(f"{name} must be > 0, got {value}")
    if not strict and not (value >= 0):
        raise ValueError(f"{name} must be >= 0, got {value}")


def _assert_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _assert_in(name: str, value: str, choices: List[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value}")


def _assert_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value)}")


def _assert_int_range(
    name: str,
    value: int,
    lo: int,
    hi: Optional[int] = None,
    inclusive: bool = True,
) -> None:
    if hi is None:
        if inclusive and not (value >= lo):
            raise ValueError(f"{name} must be >= {lo}, got {value}")
        if (not inclusive) and not (value > lo):
            raise ValueError(f"{name} must be > {lo}, got {value}")
    else:
        if inclusive:
            if not (lo <= value <= hi):
                raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
        else:
            if not (lo < value < hi):
                raise ValueError(f"{name} must be in ({lo}, {hi}), got {value}")


# ---------------------------------------------------------------------------
# 1) FGS1 Mamba SSM Encoder (temporal)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FGS1MambaConfig:
    enabled: bool = True
    in_features: int = 6
    d_model: int = 256
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 8
    bidirectional: bool = True
    dropout: float = 0.10
    layer_norm: str = "rms"
    proj_out: int = 256
    export_step_latents: bool = False
    amp: bool = True
    torchscript_safe: bool = True

    def __post_init__(self) -> None:
        _assert_bool("enabled", self.enabled)
        _assert_int_range("in_features", self.in_features, 1)
        _assert_int_range("d_model", self.d_model, 1)
        _assert_int_range("d_state", self.d_state, 1)
        _assert_int_range("d_conv", self.d_conv, 1)
        _assert_int_range("expand", self.expand, 1)
        _assert_int_range("n_layers", self.n_layers, 1)
        _assert_bool("bidirectional", self.bidirectional)
        _assert_in("layer_norm", self.layer_norm, ["layernorm", "rms"])
        _assert_int_range("proj_out", self.proj_out, 1)
        _assert_bool("export_step_latents", self.export_step_latents)
        _assert_bool("amp", self.amp)
        _assert_bool("torchscript_safe", self.torchscript_safe)
        _assert_non_negative("dropout", self.dropout)


# ---------------------------------------------------------------------------
# 2) AIRS GNN Encoder (spectral graph)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AIRSGNNConfig:
    enabled: bool = True
    in_features: int = 256
    gnn_type: str = "gat"
    num_layers: int = 4
    hidden_dim: int = 256
    heads: int = 4
    dropout: float = 0.10
    edge_features: List[str] = field(
        default_factory=lambda: [
            "delta_lambda",
            "mol_tag_i",
            "mol_tag_j",
            "seam_flag",
        ]
    )
    use_edge_attr: bool = True
    export_attention_weights: bool = True
    torchscript_safe: bool = True

    def __post_init__(self) -> None:
        _assert_bool("enabled", self.enabled)
        _assert_int_range("in_features", self.in_features, 1)
        _assert_in("gnn_type", self.gnn_type, ["gat", "gcn", "edge_gat", "nnconv"])
        _assert_int_range("num_layers", self.num_layers, 1)
        _assert_int_range("hidden_dim", self.hidden_dim, 1)
        _assert_int_range("heads", self.heads, 1)
        _assert_non_negative("dropout", self.dropout)
        if not isinstance(self.edge_features, list) or not all(
            isinstance(x, str) for x in self.edge_features
        ):
            raise ValueError("edge_features must be a list[str]")
        _assert_bool("use_edge_attr", self.use_edge_attr)
        _assert_bool("export_attention_weights", self.export_attention_weights)
        _assert_bool("torchscript_safe", self.torchscript_safe)


# ---------------------------------------------------------------------------
# 3) Multi-Scale μ Decoder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MultiScaleDecoderConfig:
    enabled: bool = True
    base_dim: int = 256
    scales: List[str] = field(default_factory=lambda: ["coarse", "mid", "fine"])
    skip_connections: bool = True
    activation: str = "gelu"
    dropout: float = 0.05
    output_bins: int = 283
    symbolic_aware: bool = True

    def __post_init__(self) -> None:
        _assert_bool("enabled", self.enabled)
        _assert_int_range("base_dim", self.base_dim, 1)
        if not self.scales or any(s not in ("coarse", "mid", "fine") for s in self.scales):
            raise ValueError('scales must be non-empty subset of {"coarse","mid","fine"}')
        _assert_bool("skip_connections", self.skip_connections)
        _assert_in("activation", self.activation, ["gelu", "relu", "silu"])
        _assert_non_negative("dropout", self.dropout)
        _assert_int_range("output_bins", self.output_bins, 1)
        _assert_bool("symbolic_aware", self.symbolic_aware)


# ---------------------------------------------------------------------------
# 4) σ Head (flow / quantile / mlp)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FlowUncertaintyHeadConfig:
    enabled: bool = True
    type: str = "flow"
    hidden_dim: int = 256
    dropout: float = 0.05
    activation: str = "gelu"
    min_sigma: float = 1e-4
    softplus: bool = True
    predict_quantiles: bool = False
    quantile_monotonicity_loss: bool = True
    torchscript_safe: bool = True

    def __post_init__(self) -> None:
        _assert_bool("enabled", self.enabled)
        _assert_in("type", self.type, ["flow", "mlp", "quantile"])
        _assert_int_range("hidden_dim", self.hidden_dim, 1)
        _assert_non_negative("dropout", self.dropout)
        _assert_in("activation", self.activation, ["gelu", "relu", "silu"])
        _assert_positive("min_sigma", self.min_sigma, strict=False)
        _assert_bool("softplus", self.softplus)
        _assert_bool("predict_quantiles", self.predict_quantiles)
        _assert_bool("quantile_monotonicity_loss", self.quantile_monotonicity_loss)
        _assert_bool("torchscript_safe", self.torchscript_safe)


# ---------------------------------------------------------------------------
# 5) Symbolic Physics Loss Pack
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SymbolicLossConfig:
    smoothness_lambda: float = 0.01
    nonnegativity_lambda: float = 0.01
    molecular_coherence_lambda: float = 0.05
    seam_continuity_lambda: float = 0.02
    chemistry_ratio_lambda: float = 0.02
    quantile_monotonicity_lambda: float = 0.01
    molecule_windows: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "H2O": [45, 67, 89],
            "CH4": [120, 140, 160],
            "CO2": [200, 220, 240],
        }
    )
    seam_index: int = 140
    chemistry_ratio_bounds: Dict[str, List[float]] = field(
        default_factory=lambda: {"CH4_H2O": [0.5, 2.0], "CO2_H2O": [0.3, 1.5]}
    )

    def __post_init__(self) -> None:
        for k in (
            "smoothness_lambda",
            "nonnegativity_lambda",
            "molecular_coherence_lambda",
            "seam_continuity_lambda",
            "chemistry_ratio_lambda",
            "quantile_monotonicity_lambda",
        ):
            _assert_non_negative(k, getattr(self, k))

        if not isinstance(self.molecule_windows, dict):
            raise ValueError("molecule_windows must be dict[str, list[int]]")
        for mol, idxs in self.molecule_windows.items():
            if not isinstance(mol, str):
                raise ValueError("molecule_windows keys must be str")
            if not isinstance(idxs, list) or not all(isinstance(i, int) and i >= 0 for i in idxs):
                raise ValueError(f"molecule_windows[{mol}] must be list[int >=0]")
            if len(idxs) == 0:
                pass  # tolerate empty windows

        _assert_int_range("seam_index", self.seam_index, 0, 282)

        if not isinstance(self.chemistry_ratio_bounds, dict):
            raise ValueError(
                "chemistry_ratio_bounds must be dict[str, (float, float)]"
            )
        for ratio, bounds in self.chemistry_ratio_bounds.items():
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or not all(isinstance(x, (int, float)) for x in bounds)
            ):
                raise ValueError(
                    f"chemistry_ratio_bounds[{ratio}] must be [float_min, float_max]"
                )
            lo, hi = float(bounds[0]), float(bounds[1])
            if not (0.0 <= lo < hi):
                raise ValueError(
                    f"chemistry_ratio_bounds[{ratio}] invalid bounds: ({lo}, {hi}); require 0 <= lo < hi"
                )


# ---------------------------------------------------------------------------
# 6) Fusion Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FusionConfig:
    type: str = "concat_mlp"
    hidden_dim: int = 256
    dropout: float = 0.05
    gate_init_bias: float = 0.0
    symbolic_aware: bool = True

    def __post_init__(self) -> None:
        _assert_in("type", self.type, ["concat_mlp", "cross_attention", "gated_fusion"])
        _assert_int_range("hidden_dim", self.hidden_dim, 1)
        _assert_non_negative("dropout", self.dropout)
        _assert_bool("symbolic_aware", self.symbolic_aware)


# ---------------------------------------------------------------------------
# 7) Composed envelope for "spectramind" model/src namespace
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SpectraMindModelSrcConfig:
    """Top-level envelope mirroring configs/model/src/spectramind/__init__.yaml."""

    fgs1_mamba: FGS1MambaConfig = field(default_factory=FGS1MambaConfig)
    airs_gnn: AIRSGNNConfig = field(default_factory=AIRSGNNConfig)
    multi_scale_decoder: MultiScaleDecoderConfig = field(
        default_factory=MultiScaleDecoderConfig
    )
    flow_uncertainty_head: FlowUncertaintyHeadConfig = field(
        default_factory=FlowUncertaintyHeadConfig
    )
    symbolic_loss: SymbolicLossConfig = field(default_factory=SymbolicLossConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def as_dict(self) -> Dict[str, dict]:
        return OmegaConf.to_container(OmegaConf.create(self), resolve=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hydra ConfigStore registration convenience
# ---------------------------------------------------------------------------

def register_spectramind_config_schemas(
    group: str = "model/src/spectramind",
    name_prefix: str = "schema_",
) -> None:
    """Register schemas in Hydra's ConfigStore for @hydra.main defaults."""
    cs = ConfigStore.instance()

    cs.store(group=group, name=f"{name_prefix}fgs1_mamba", node=FGS1MambaConfig)
    cs.store(group=group, name=f"{name_prefix}airs_gnn", node=AIRSGNNConfig)
    cs.store(
        group=group,
        name=f"{name_prefix}multi_scale_decoder",
        node=MultiScaleDecoderConfig,
    )
    cs.store(
        group=group,
        name=f"{name_prefix}flow_uncertainty_head",
        node=FlowUncertaintyHeadConfig,
    )
    cs.store(
        group=group, name=f"{name_prefix}symbolic_loss", node=SymbolicLossConfig
    )
    cs.store(group=group, name=f"{name_prefix}fusion", node=FusionConfig)
    cs.store(group=group, name=f"{name_prefix}all", node=SpectraMindModelSrcConfig)


# ---------------------------------------------------------------------------
# Standalone validation/merge of an arbitrary YAML file or text
# ---------------------------------------------------------------------------

def load_and_validate_yaml(
    yaml_path_or_str: str,
    expect_root_key: Optional[str] = "spectramind",
    compose_into: str = "model/src/spectramind/schema_all",  # for future use
) -> DictConfig:
    """Load YAML, merge with structured schema, and validate via dataclasses."""
    _ = compose_into  # unused for now but kept for API compatibility
    register_spectramind_config_schemas()
    base = OmegaConf.structured(SpectraMindModelSrcConfig)  # type: ignore[arg-type]

    if os.path.exists(yaml_path_or_str) and os.path.isfile(yaml_path_or_str):
        user_cfg = OmegaConf.load(yaml_path_or_str)
    else:
        user_cfg = OmegaConf.create(yaml_path_or_str)

    if expect_root_key is not None:
        if expect_root_key not in user_cfg:
            raise ValueError(
                f"Expected root key '{expect_root_key}' in YAML; got top-level keys: {list(user_cfg.keys())}"
            )

    merged = OmegaConf.merge(
        base, user_cfg.get(expect_root_key) if expect_root_key else user_cfg
    )

    _ = SpectraMindModelSrcConfig(
        fgs1_mamba=FGS1MambaConfig(
            **OmegaConf.to_container(merged.get("fgs1_mamba"), resolve=True)
        ),
        airs_gnn=AIRSGNNConfig(
            **OmegaConf.to_container(merged.get("airs_gnn"), resolve=True)
        ),
        multi_scale_decoder=MultiScaleDecoderConfig(
            **OmegaConf.to_container(merged.get("multi_scale_decoder"), resolve=True)
        ),
        flow_uncertainty_head=FlowUncertaintyHeadConfig(
            **OmegaConf.to_container(merged.get("flow_uncertainty_head"), resolve=True)
        ),
        symbolic_loss=SymbolicLossConfig(
            **OmegaConf.to_container(merged.get("symbolic_loss"), resolve=True)
        ),
        fusion=FusionConfig(
            **OmegaConf.to_container(merged.get("fusion"), resolve=True)
        ),
    )

    if expect_root_key:
        return OmegaConf.create({expect_root_key: merged})
    return merged


if __name__ == "__main__":
    demo = """
    spectramind:
      fgs1_mamba:
        enabled: true
        in_features: 6
      airs_gnn:
        gnn_type: gat
      multi_scale_decoder:
        output_bins: 283
      flow_uncertainty_head:
        type: flow
        min_sigma: 0.0001
      symbolic_loss:
        seam_index: 140
      fusion:
        type: concat_mlp
    """
    cfg = load_and_validate_yaml(demo)
    print(OmegaConf.to_yaml(cfg, resolve=True))
