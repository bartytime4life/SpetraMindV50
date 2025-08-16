from __future__ import annotations

import ast
import re
from typing import Any, Dict

_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _coerce_val(v: str) -> Any:
    """
    Attempt to coerce a string into bool/int/float/list/dict via ast.literal_eval,
    falling back to string on failure.
    """
    # Fast-path number
    if _NUMBER_RE.match(v):
        return float(v) if "." in v else int(v)
    # Booleans / None / lists / dicts etc.
    try:
        lit = ast.literal_eval(v)
        return lit
    except Exception:
        # pass-through strings like "fgs1_mamba"
        return v


def cli_override_parser(override_list: list[str]) -> Dict[str, Any]:
    """
    Parse CLI override strings into a flattened dictionary.

    Example:
        ["train.lr=0.001", "model=fgs1_mamba", "flags.debug=True"]
        -> {"train.lr": 0.001, "model": "fgs1_mamba", "flags.debug": True}
    """
    parsed: Dict[str, Any] = {}
    for o in override_list:
        if "=" not in o:
            raise ValueError(f"Invalid override (missing '='): {o}")
        k, v = o.split("=", 1)
        parsed[k.strip()] = _coerce_val(v.strip())
    return parsed


def apply_overrides(cfg, override_dict: Dict[str, Any]):
    """
    Apply overrides onto an OmegaConf object using dot-keys.
    """
    for k, v in override_dict.items():
        cfg[k] = v
    return cfg
