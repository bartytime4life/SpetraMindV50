from __future__ import annotations

import json
import os
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def safe_get(dct: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safely retrieve a dotted path from a dictionary."""
    cur: Any = dct
    for token in path.split("."):
        if not isinstance(cur, dict) or token not in cur:
            return default
        cur = cur[token]
    return cur


def write_run_manifest(path: str, payload: Dict[str, Any]) -> None:
    """Write a JSON manifest to ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
