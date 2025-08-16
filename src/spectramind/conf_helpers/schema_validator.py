"""Validate configurations against a JSON schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from jsonschema import ValidationError, validate
from omegaconf import DictConfig, OmegaConf


def validate_config(
    cfg: Union[Dict[str, Any], DictConfig], schema_path: str | Path
) -> bool:
    """Validate ``cfg`` against the JSON schema at ``schema_path``.

    Parameters
    ----------
    cfg:
        Configuration to validate. May be a plain ``dict`` or ``DictConfig``.
    schema_path:
        Path to a JSON schema file.

    Returns
    -------
    bool
        ``True`` if the config conforms to the schema, ``False`` otherwise.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    with open(Path(schema_path), "r", encoding="utf-8") as f:
        schema = json.load(f)

    try:
        validate(instance=cfg, schema=schema)
    except ValidationError:
        return False
    return True
