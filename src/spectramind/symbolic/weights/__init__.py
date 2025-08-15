"""SpectraMind V50 — Symbolic Weights Subsystem.

Exports: loading, validation, optimization, profiles, CLI app hook.
This package is Hydra-safe (pure-Python YAML loading), logs to console and rotating file,
and emits a JSONL event stream for reproducibility.
"""

from .auto_weight_optimizer import (
    OptimizationConfig,
    apply_metric_driven_adjustments,
    optimize_symbolic_weights,
)
from .cli import app as cli_app
from .loader import (
    compose_weights,
    list_available_weight_sets,
    load_all_known_weights,
    load_symbolic_weights,
)
from .validator import (
    WeightProfileSchema,
    WeightSchema,
    validate_profile_weights,
    validate_weight_config,
)

__all__ = [
    "load_symbolic_weights",
    "list_available_weight_sets",
    "compose_weights",
    "load_all_known_weights",
    "WeightSchema",
    "WeightProfileSchema",
    "validate_weight_config",
    "validate_profile_weights",
    "optimize_symbolic_weights",
    "OptimizationConfig",
    "apply_metric_driven_adjustments",
    "cli_app",
]
