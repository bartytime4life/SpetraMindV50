"""
SpectraMind V50 Config Package
Provides loading, validation, registry utilities, and auto-registration of default configs.

Master-programmer features:
- On import, automatically registers:
  - 'defaults' from defaults.yaml
  - 'v50' from config_v50.yaml
- Safe, idempotent, Hydra-compatible, schema-validated.
- Logs lightweight status to stdout only on first import (can be silenced via SPECTRAMIND_CONFIG_SILENT=1).

Directory layout expected beside this file:
- base_config.py, config_loader.py, config_validator.py, config_schema.yaml
- defaults.yaml, config_v50.yaml, registry.py, hydra_overrides.yaml, env.py
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any

from .base_config import BaseConfig
from .config_loader import load_config, load_hydra_config
from .config_validator import validate_config
from .registry import CONFIG_REGISTRY, register_config

__all__ = [
    "BaseConfig",
    "load_config",
    "load_hydra_config",
    "validate_config",
    "CONFIG_REGISTRY",
    "register_config",
]

# ---- Auto-registration on import -------------------------------------------------

def _maybe_print(msg: str) -> None:
    # Respect a simple env toggle to silence messages in CI/production
    if os.environ.get("SPECTRAMIND_CONFIG_SILENT", "0") not in ("1", "true", "True"):
        print(msg)

def _safe_register(name: str, cfg: Dict[str, Any]) -> None:
    # Idempotent registration (won't overwrite an existing key)
    if name not in CONFIG_REGISTRY:
        register_config(name, cfg)

def _load_local_yaml(rel_filename: str) -> Dict[str, Any]:
    # Load a YAML config located next to this package, with schema validation
    here = Path(__file__).parent
    cfg_path = here / rel_filename
    if not cfg_path.exists():
        raise FileNotFoundError(f"[spectramind.config] Missing expected file: {cfg_path}")
    cfg = load_config(str(cfg_path))
    return cfg

def _auto_register_defaults() -> None:
    try:
        defaults_cfg = _load_local_yaml("defaults.yaml")
        _safe_register("defaults", defaults_cfg)
    except Exception as e:
        _maybe_print(f"[spectramind.config] defaults auto-register skipped: {e}")

def _auto_register_v50() -> None:
    try:
        v50_cfg = _load_local_yaml("config_v50.yaml")
        _safe_register("v50", v50_cfg)
    except Exception as e:
        _maybe_print(f"[spectramind.config] v50 auto-register skipped: {e}")

# Perform auto-registration once per interpreter session
try:
    _auto_register_defaults()
    _auto_register_v50()
    _maybe_print("[spectramind.config] Auto-registered configs: "
                 + ", ".join(sorted(list(CONFIG_REGISTRY.keys()))))
except Exception as _e:
    _maybe_print(f"[spectramind.config] Auto-registration encountered an issue: {_e}")
