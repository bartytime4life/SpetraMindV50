# SPDX-License-Identifier: MIT

"""Tools for applying symbolic overrides to configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Union
from omegaconf import OmegaConf, DictConfig

from .io import load_yaml
from .logging_utils import write_md, log_event
from .exceptions import OverrideLoadError


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionary ``u`` into ``d``."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d


def _to_dict(cfg: Union[Dict[str, Any], DictConfig]) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def apply_symbolic_overrides(
    config: Union[Dict[str, Any], DictConfig],
    override_path: str | Path,
) -> Union[Dict[str, Any], DictConfig]:
    """Apply a single YAML override file to a configuration."""
    try:
        ovr = load_yaml(override_path)
        write_md("Applying symbolic overrides", {"override_path": str(override_path)})
        log_event("apply_override", {"override_path": str(override_path)})
        base = _to_dict(config)
        merged = deep_update(base, ovr)
        return OmegaConf.create(merged) if isinstance(config, DictConfig) else merged
    except Exception as e:  # pragma: no cover - YAML issues
        write_md("Apply override failed", {"error": str(e), "override_path": str(override_path)})
        log_event("apply_override_error", {"error": str(e)})
        raise OverrideLoadError(str(e)) from e


def load_overrides_layered(
    config: Union[Dict[str, Any], DictConfig],
    paths: List[str | Path],
) -> Union[Dict[str, Any], DictConfig]:
    """Apply a sequence of override files in order."""
    out = config
    for p in paths:
        out = apply_symbolic_overrides(out, p)
    return out
