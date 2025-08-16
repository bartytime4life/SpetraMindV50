from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from omegaconf import OmegaConf


def _load_schema(schema_path: str) -> Any:
    p = Path(schema_path)
    if not p.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    text = p.read_text()
    try:
        # Try JSON first
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to YAML
        return yaml.safe_load(text)


def validate_config(cfg, schema_path: str) -> bool:
    """
    Validate a config (OmegaConf or dict) against a JSON/YAML JSON-Schema.

    Raises:
        jsonschema.ValidationError on failure.
    """
    schema = _load_schema(schema_path)
    cfg_dict = (
        OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    )
    jsonschema.validate(cfg_dict, schema)
    return True
