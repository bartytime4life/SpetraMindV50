"""
Copyright (c) 2025 SpectraMind.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any

from .base_config import Config


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_config(cfg: Config) -> None:
    """
    Multi-layer validation:
    1) Type/shape semantics via dataclasses (done on construction).
    2) Cross-field constraints and numeric ranges.
    3) Mission-aware bounds (sanity guards).
    Raises ValueError with actionable messages.
    """
    c = asdict(cfg)

    # Device check
    _assert(cfg.training.device in {"cuda", "cpu"}, "training.device must be in {'cuda','cpu'}")

    # Learning rate and epochs
    _assert(cfg.training.learning_rate > 0.0, "training.learning_rate must be > 0")
    _assert(1 <= cfg.training.epochs <= 10_000, "training.epochs must be within [1, 10000]")

    # Batch/grad accumulation
    _assert(cfg.training.batch_size >= 1, "training.batch_size must be >= 1")
    _assert(cfg.training.grad_accum >= 1, "training.grad_accum must be >= 1")

    # Symbolic weights bounds
    for k, w in cfg.model.symbolic_loss_weights.items():
        _assert(0.0 <= float(w) <= 10.0, f"model.symbolic_loss_weights['{k}'] must be in [0, 10]")

    # Conformal coverage
    if cfg.model.conformal_corel:
        _assert(0.5 <= cfg.model.corel_coverage < 1.0, "model.corel_coverage must be in [0.5, 1.0)")

    # Paths exist or are creatable (non-empty)
    for p_name in ["runs_dir", "logs_dir", "artifacts_dir", "reports_dir", "checkpoints_dir"]:
        p = getattr(cfg.paths, p_name, None)
        _assert(p is not None, f"paths.{p_name} must be set")

    # Logging rotation
    _assert(cfg.logging.rotating_bytes >= 1024, "logging.rotating_bytes must be >= 1 KiB")
    _assert(0 <= cfg.logging.rotating_backups <= 1000, "logging.rotating_backups must be in [0,1000]")

    # Data augmentations sanity
    aug: Dict[str, Any] = cfg.data.augmentations
    if "temporal_dropout" in aug:
        _assert(0.0 <= float(aug["temporal_dropout"]) < 1.0, "data.augmentations.temporal_dropout must be in [0,1)")
    if "airs_noise_sigma" in aug:
        _assert(0.0 <= float(aug["airs_noise_sigma"]) <= 0.1, "data.augmentations.airs_noise_sigma must be <= 0.1")

    # Precision mode check
    _assert(cfg.training.precision in {"fp32", "bf16", "amp"}, "training.precision must be one of {'fp32','bf16','amp'}")

    # Optional MLflow/W&B toggles consistent
    if cfg.training.mlflow_enable:
        pass  # tracking URI may be None; runtime can fall back to env
    if cfg.training.wandb_enable:
        _assert(bool(cfg.training.wandb_project), "training.wandb_project must be set when wandb_enable=True")
