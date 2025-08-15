"""Hydra-safe loader for symbolic configuration files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .symbolic_utils import append_debug_log, hash_config_like, now_utc_iso, read_yaml

DEFAULT_RULES = "src/spectramind/symbolic/rules/default_rules.yaml"
DEFAULT_WEIGHTS = "src/spectramind/symbolic/weights/default_weights.yaml"
DEFAULT_PROFILES = "src/spectramind/symbolic/profiles/default_profiles.yaml"
DEBUG_LOG = "logs/v50_debug_log.md"


@dataclass
class SymbolicConfig:
    rules_path: str = DEFAULT_RULES
    weights_path: str = DEFAULT_WEIGHTS
    profiles_path: str = DEFAULT_PROFILES
    overrides_dir: Optional[str] = "src/spectramind/symbolic/overrides"
    seed: int = 42
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rules_path": self.rules_path,
            "weights_path": self.weights_path,
            "profiles_path": self.profiles_path,
            "overrides_dir": self.overrides_dir,
            "seed": self.seed,
            "meta": self.meta,
        }


def _first_existing(paths: list[str]) -> Optional[str]:
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


def load_symbolic_config(
    rules_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    profiles_path: Optional[str] = None,
    overrides_dir: Optional[str] = None,
    seed: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> SymbolicConfig:
    """Load SymbolicConfig with fallbacks to defaults and env overrides."""
    rp = rules_path or os.environ.get("SYMBOLIC_RULES", DEFAULT_RULES)
    wp = weights_path or os.environ.get("SYMBOLIC_WEIGHTS", DEFAULT_WEIGHTS)
    pp = profiles_path or os.environ.get("SYMBOLIC_PROFILES", DEFAULT_PROFILES)
    od = overrides_dir or os.environ.get(
        "SYMBOLIC_OVERRIDES", "src/spectramind/symbolic/overrides"
    )
    sd = seed if seed is not None else int(os.environ.get("SYMBOLIC_SEED", "42"))
    mt = meta or {}

    # Validate existence with graceful log
    missing = []
    for label, p in [("rules", rp), ("weights", wp), ("profiles", pp)]:
        if not Path(p).exists():
            missing.append((label, p))
    if missing:
        msg = (
            f"[{now_utc_iso()}] symbolic_config_loader: Missing config files: {missing}"
        )
        append_debug_log(DEBUG_LOG, msg)

    cfg = SymbolicConfig(
        rules_path=rp,
        weights_path=wp,
        profiles_path=pp,
        overrides_dir=od,
        seed=sd,
        meta=mt,
    )

    stamp = now_utc_iso()
    run_hash = hash_config_like(cfg.to_dict())
    append_debug_log(
        DEBUG_LOG,
        f"[{stamp}] load_symbolic_config: hash={run_hash} rules={rp} weights={wp} profiles={pp} overrides={od} seed={sd}",
    )
    return cfg


def load_yaml_checked(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        return {}
    data = read_yaml(path)
    return data if isinstance(data, dict) else {}
