"""Unified loader for symbolic weights (YAML), with composition logic.

Features:
    • Deterministic YAML loading (yaml.safe_load) and stable merges
    • Weight composition: base + penalties + molecule-region + profile overrides
    • Discovery helpers and manifest hashing for reproducibility
    • Structured logging and JSONL event emission
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .logging_utils import emit_event, get_logger

LOGGER = get_logger(__name__)
PKG_DIR = Path(__file__).resolve().parent

# Default YAML files in this package; these can be overridden by Hydra/config paths if desired.
DEFAULT_WEIGHT_FILES = [
    "base_weights.yaml",
    "smoothness_penalty.yaml",
    "nonnegativity_penalty.yaml",
    "seam_continuity_penalty.yaml",
    "quantile_monotonicity.yaml",
    "molecule_region_weights.yaml",
]


def _read_yaml(p: Path) -> Dict[str, Any]:
    """Read a YAML file safely and return a dict. Raises on missing file or parse error."""
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML must map to dict at top-level: {p}")
    return data


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge src into dst (in place) with right-precedence.

    Numeric leaves overwrite; dicts recurse; lists overwrite (right wins).
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def list_available_weight_sets(directory: Path | None = None) -> List[str]:
    """List .yaml weight files in the weights directory (no profiles)."""
    d = directory or PKG_DIR
    names: List[str] = []
    for p in sorted(d.glob("*.yaml")):
        names.append(p.name)
    return names


def load_symbolic_weights(
    filename: str, directory: Path | None = None
) -> Dict[str, Any]:
    """Load a single weight YAML by filename from given directory (defaults to package dir)."""
    d = directory or PKG_DIR
    cfg = _read_yaml(d / filename)
    LOGGER.info(
        "loaded_weight_file",
        extra={"file": str(d / filename), "keys": list(cfg.keys())},
    )
    emit_event("weights_loaded", {"file": str(d / filename), "keys": list(cfg.keys())})
    return cfg


def load_all_known_weights(
    files: List[str] | None = None, directory: Path | None = None
) -> List[Tuple[str, Dict[str, Any]]]:
    """Load all specified YAML weight files (or defaults) and return (filename, dict) list."""
    d = directory or PKG_DIR
    names = files or DEFAULT_WEIGHT_FILES
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name in names:
        out.append((name, load_symbolic_weights(name, d)))
    return out


def compose_weights(
    base_files: List[str] | None = None,
    directory: Path | None = None,
    profile_file: str | None = None,
    profile_name: str | None = None,
) -> Dict[str, Any]:
    """Compose final weights in precedence order:

    1) base/penalties/molecule-region defaults
    2) composite_profile_weights.yaml[profile_name] (if provided)
    3) explicit profile YAML file (if provided)

    Returns a flat dict of weights usable by symbolic_loss and friends.
    """
    d = directory or PKG_DIR
    merged: Dict[str, Any] = {}

    # Step 1: defaults
    for fname, cfg in load_all_known_weights(base_files, d):
        LOGGER.debug("merge_weight_file", extra={"file": fname})
        _deep_merge(merged, cfg)

    # Step 2: composite profile
    if profile_name:
        comp = _read_yaml(d / "composite_profile_weights.yaml")
        if profile_name not in comp:
            raise KeyError(
                f"Profile '{profile_name}' not found in composite_profile_weights.yaml"
            )
        LOGGER.info("apply_composite_profile", extra={"profile": profile_name})
        _deep_merge(merged, comp[profile_name])

    # Step 3: explicit profile file (e.g., profiles/hot_jupiter.yaml)
    if profile_file:
        p = (
            d / "profiles" / profile_file
            if not profile_file.startswith("/")
            else Path(profile_file)
        )
        LOGGER.info("apply_profile_file", extra={"profile_file": str(p)})
        prof_dict = _read_yaml(p)
        _deep_merge(merged, prof_dict)

    manifest = _manifest_for_config(merged)
    LOGGER.info(
        "compose_weights_done",
        extra={"num_keys": len(merged), "hash": manifest.config_hash},
    )
    emit_event(
        "weights_composed", {"num_keys": len(merged), "hash": manifest.config_hash}
    )
    return merged


@dataclass(frozen=True)
class Manifest:
    files: List[str]
    config_hash: str


def _manifest_for_config(
    cfg: Dict[str, Any], files: List[str] | None = None
) -> Manifest:
    """Compute a stable SHA256 hash for the composed config; record contributing files if provided."""
    buf = io.BytesIO(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    h = hashlib.sha256(buf.getvalue()).hexdigest()
    return Manifest(files=files or list_available_weight_sets(), config_hash=h)
