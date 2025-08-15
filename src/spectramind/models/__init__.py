# Copyright (c) 2025.
# SpectraMind V50 - Models Package
# This __init__ flattens imports and exposes a stable registry for CLI/config loading.

from .airs_gnn import AIRSGNN
from .fgs1_mamba import FGS1MambaEncoder
from .multi_scale_decoder import MultiScaleDecoder
from .flow_uncertainty_head import FlowUncertaintyHead
from .symbolic_logic_engine import SymbolicLogicEngine
from .symbolic_loss import SymbolicLoss, SymbolicLossConfig, symbolic_loss_from_yaml
from .symbolic_violation_predictor import SymbolicViolationPredictor
from .symbolic_violation_predictor_nn import SymbolicViolationPredictorNN
from .symbolic_fusion_predictor import SymbolicFusionPredictor
from .fusion_attention_diagnostics import FusionAttentionDiagnostics, AttentionTrace
from .neural_logic_graph import NeuralLogicGraph
from .model_utils import (
    init_weights,
    count_parameters,
    summarize_module,
    save_checkpoint,
    load_checkpoint,
    seed_everything,
)

__all__ = [
    "AIRSGNN",
    "FGS1MambaEncoder",
    "MultiScaleDecoder",
    "FlowUncertaintyHead",
    "SymbolicLogicEngine",
    "SymbolicLoss",
    "SymbolicLossConfig",
    "symbolic_loss_from_yaml",
    "SymbolicViolationPredictor",
    "SymbolicViolationPredictorNN",
    "SymbolicFusionPredictor",
    "FusionAttentionDiagnostics",
    "AttentionTrace",
    "NeuralLogicGraph",
    "init_weights",
    "count_parameters",
    "summarize_module",
    "save_checkpoint",
    "load_checkpoint",
    "seed_everything",
]
