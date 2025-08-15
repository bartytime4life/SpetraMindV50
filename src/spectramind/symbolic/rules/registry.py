# SPDX-License-Identifier: Apache-2.0

"""Lightweight registry for symbolic rules."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type

from .base_rule import SymbolicRule
from .utils import get_logger

RULE_REGISTRY: Dict[str, Type[SymbolicRule]] = {}
_log = get_logger("spectramind.symbolic.rules.registry")


def register_rule(name: str):
    """Decorator to register a ``SymbolicRule`` subclass with a given name."""

    def _wrap(cls: Type[SymbolicRule]) -> Type[SymbolicRule]:
        if name in RULE_REGISTRY:
            _log.warning("Overwriting registration for rule %s", name)
        RULE_REGISTRY[name] = cls
        return cls

    return _wrap


def auto_discover(package: str = "src.spectramind.symbolic.rules") -> None:
    """Import all modules in the rules package so ``@register_rule`` runs."""

    try:
        pkg = importlib.import_module(package)
        for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            if not ispkg:
                try:
                    importlib.import_module(modname)
                except Exception as exc:  # pragma: no cover - discovery is best effort
                    _log.warning("Failed to import %s: %s", modname, exc)
    except Exception as exc:  # pragma: no cover - discovery optional
        _log.warning("Auto-discover skipped: %s", exc)


def get_rule(name: str) -> Optional[Type[SymbolicRule]]:
    return RULE_REGISTRY.get(name)


def list_rules() -> List[str]:
    return sorted(RULE_REGISTRY.keys())


def build_rule_from_config(cfg: Dict[str, Any]) -> SymbolicRule:
    """Build a rule from a config dict."""

    assert isinstance(cfg, dict) and "name" in cfg, "Config must include 'name'"
    name = str(cfg["name"])
    params = dict(cfg.get("params", {}))
    cls = get_rule(name)
    if cls is None:
        raise KeyError(f"Unknown rule '{name}'. Available: {list_rules()}")
    return cls(**params)


__all__ = [
    "RULE_REGISTRY",
    "register_rule",
    "auto_discover",
    "get_rule",
    "list_rules",
    "build_rule_from_config",
]
