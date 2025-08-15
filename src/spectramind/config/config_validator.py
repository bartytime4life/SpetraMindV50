"""
Schema-based config validator for SpectraMind V50 configs.

Uses jsonschema Draft-07 via a YAML schema file located next to this module.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import yaml
from jsonschema import validate, ValidationError

SCHEMA_PATH = Path(__file__).parent / "config_schema.yaml"

def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validate a config dict against the local JSON schema.

    Raises:
        FileNotFoundError: If the schema file is missing.
        ValueError: If validation fails with details.
    """
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Config schema not found at {SCHEMA_PATH}")
    with open(SCHEMA_PATH, "r") as f:
        schema = yaml.safe_load(f) or {}
    try:
        validate(instance=cfg, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e.message}") from e
