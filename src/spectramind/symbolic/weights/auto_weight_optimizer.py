"""Metric-driven symbolic weight optimization."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Tuple

from .logging_utils import emit_event, get_logger

LOGGER = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Tunable knobs for optimization."""

    target: float = 1.0
    step: float = 0.10
    min_delta: float = -0.25
    max_delta: float = 0.25
    absolute_clip_min: float = 0.0
    absolute_clip_max: float = 10.0
    dry_run: bool = False


def _bounded_adjust(cur_w: float, error: float, cfg: OptimizationConfig) -> float:
    """Compute a bounded relative adjustment based on error = (target - metric).

    Positive error -> increase weight; negative -> decrease.
    """
    # Proportional term with a smooth squashing for large errors.
    raw_rel = cfg.step * math.tanh(error)
    rel = max(cfg.min_delta, min(cfg.max_delta, raw_rel))
    new_val = cur_w * (1.0 + rel)
    return max(cfg.absolute_clip_min, min(cfg.absolute_clip_max, new_val))


def apply_metric_driven_adjustments(
    weights: Dict[str, float],
    metrics: Dict[str, float],
    cfg: OptimizationConfig,
) -> Dict[str, float]:
    updated = copy.deepcopy(weights)
    changes: Dict[str, Tuple[float, float]] = {}
    for rule, cur_w in weights.items():
        if not isinstance(cur_w, (int, float)):
            continue
        if rule not in metrics:
            continue
        m = metrics[rule]
        error = cfg.target - m
        new_w = _bounded_adjust(float(cur_w), error, cfg)
        if cfg.dry_run:
            # Do not mutate; just record preview
            changes[rule] = (float(cur_w), new_w)
        else:
            updated[rule] = new_w
            changes[rule] = (float(cur_w), new_w)

    LOGGER.info("weight_adjustments", extra={"num_rules": len(changes)})
    emit_event(
        "weights_adjusted",
        {
            "changes": {k: {"old": a, "new": b} for k, (a, b) in changes.items()},
            "dry_run": cfg.dry_run,
        },
    )
    return updated


def optimize_symbolic_weights(
    base_weights: Dict[str, float],
    performance_metrics: Dict[str, float] | None = None,
    cfg: OptimizationConfig | None = None,
) -> Dict[str, float]:
    """Optimize weights given performance metrics; if metrics is None, returns base unchanged."""
    if not performance_metrics:
        LOGGER.warning(
            "optimize_no_metrics",
            extra={"message": "No metrics provided; returning base weights"},
        )
        emit_event("optimize_skipped", {"reason": "no_metrics"})
        return dict(base_weights)
    ocfg = cfg or OptimizationConfig()
    return apply_metric_driven_adjustments(base_weights, performance_metrics, ocfg)
