"""
Global config registry for SpectraMind V50.

Simple, explicit mapping name -> config dict.
Idempotent semantics are enforced by __init__ import hooks.
"""

from typing import Dict, Any

CONFIG_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_config(name: str, cfg: Dict[str, Any]) -> None:
    """
    Register a config dict under a unique name.

    Note:
      - Overwrites are intentionally disallowed via higher-level helpers.
      - Use spectramind.config.register_config directly if you do want to overwrite.
    """
    CONFIG_REGISTRY[name] = cfg

def get_config(name: str) -> Dict[str, Any]:
    """
    Retrieve a registered config by name.

    Raises:
        KeyError: If name not present.
    """
    if name not in CONFIG_REGISTRY:
        raise KeyError(f"Config '{name}' not found in registry")
    return CONFIG_REGISTRY[name]
