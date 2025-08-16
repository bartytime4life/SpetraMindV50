# src/spectramind/conf_helpers/profile_wire.py
# Purpose: one-call “wire-in” for active profile across CLIs: set seeds, attach
# profile context to logs, and print a compact profile banner.
#
# Usage in any CLI entrypoint (right after you have `cfg` from Hydra):
#   from spectramind.conf_helpers.profile_wire import apply_profile_wire
#   cfg = apply_profile_wire(cfg)
#
# This keeps all profile glue centralized and avoids duplicating logging/seed logic.

from __future__ import annotations
import os
import random
import time
from typing import Any

def _set_seeds(seed: int, deterministic: bool) -> None:
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import torch
    except Exception:
        torch = None

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]

def _ensure_dirs(cfg: Any) -> None:
    # Create log dirs that embed the active profile if configured that way
    try:
        text_log = cfg.logging.files.text_log
        jsonl_log = cfg.logging.files.jsonl_log
        for p in (text_log, jsonl_log):
            d = os.path.dirname(str(p))
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
    except Exception:
        pass

def _log_profile_banner(cfg: Any) -> None:
    ap = getattr(cfg, "active_profile", "unknown")
    proj = getattr(getattr(cfg, "project", object()), "name", "SpectraMind")
    mode = getattr(getattr(cfg, "project", object()), "mode", "standard")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    banner = (
        f"[{ts}] [{proj}] profile={ap} mode={mode} "
        f"seed={getattr(getattr(cfg, 'reproducibility', object()), 'seed', 42)} "
        f"deterministic={getattr(getattr(cfg, 'reproducibility', object()), 'deterministic', True)}"
    )
    print(banner, flush=True)

def _attach_run_context(cfg: Any) -> None:
    # Optionally enrich JSONL event stream with profile context if your logger reads env
    os.environ["SPECTRAMIND_ACTIVE_PROFILE"] = str(getattr(cfg, "active_profile", "unknown"))
    os.environ["SPECTRAMIND_MODE"] = str(getattr(getattr(cfg, "project", object()), "mode", "standard"))
    os.environ["SPECTRAMIND_PROJECT"] = str(getattr(getattr(cfg, "project", object()), "name", "SpectraMind"))

def apply_profile_wire(cfg: Any) -> Any:
    """
    Apply active-profile wiring:
      1) set seeds & determinism
      2) ensure log directories exist (with profile in path if configured)
      3) print a compact run banner for CLI and CI logs
      4) attach env context for downstream loggers
    Returns the same cfg for fluent use.
    """
    try:
        seed = int(cfg.reproducibility.seed)
        deterministic = bool(cfg.reproducibility.deterministic)
    except Exception:
        seed, deterministic = 42, True

    _set_seeds(seed, deterministic)
    _ensure_dirs(cfg)
    _log_profile_banner(cfg)
    _attach_run_context(cfg)
    return cfg
