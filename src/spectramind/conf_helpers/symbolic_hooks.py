"""Helpers for injecting default symbolic constraint weights."""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


def inject_symbolic_constraints(cfg: DictConfig) -> DictConfig:
    """Ensure a ``symbolic.constraints`` section exists with default weights."""
    data: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    symbolic = data.setdefault("symbolic", {})
    constraints = symbolic.setdefault("constraints", {})
    constraints.setdefault("default_weight", 1.0)
    return OmegaConf.create(data)
