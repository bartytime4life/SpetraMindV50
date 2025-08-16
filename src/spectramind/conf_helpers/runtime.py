from __future__ import annotations

"""
SpectraMind V50 — Hydra runtime wiring helpers.

This module centralizes how the CLI injects a Hydra group override for runtime=<name>,
so that any command (train, predict, diagnose, submit, etc.) can honor a user-chosen
execution context like: local, kaggle, hpc, docker, ci.

Design goals:
    • Single source of truth for a “runtime” override that is guaranteed to be applied.
    • Zero-conf for callers: call compose_with_runtime() and pass optional extra overrides.
    • Deterministic and reproducible: logs the chosen runtime and the absolute config path used.
    • Backward compatible: if no runtime is provided via CLI or env, falls back to “default”.

Usage:
from spectramind.conf_helpers.runtime import (
    determine_runtime,
    compose_with_runtime,
    log_runtime_choice,
)

runtime = determine_runtime(cli_runtime)  # cli_runtime may be None
cfg = compose_with_runtime(runtime, overrides=["model=v50/fgs1_mamba", "calibration=corel"])
log_runtime_choice(runtime, cfg)

Hydra configuration requirements:
    • A config group exists at configs/runtime with files: default.yaml, local.yaml, kaggle.yaml, hpc.yaml, docker.yaml, ci.yaml
    • The CLI should pass the override runtime=<name> whenever it composes a config.
"""

import os
import pathlib
import datetime
from typing import Iterable, List, Optional

from omegaconf import DictConfig, OmegaConf

# Hydra is imported lazily inside compose_with_runtime to avoid global initialization side effects.

# Environment variable used to propagate runtime across sub-commands when the root CLI
# sets it once (e.g., spectramind --runtime kaggle submit make-submission).
_ENV_RUNTIME = "SPECTRAMIND_RUNTIME"


def determine_runtime(cli_value: Optional[str]) -> str:
    """
    Resolve the runtime name from (in order of precedence):
    1) Explicit CLI value (if provided and non-empty)
    2) Environment variable SPECTRAMIND_RUNTIME (if set)
    3) Fallback to "default"

    Parameters
    ----------
    cli_value : Optional[str]
        A string passed from the CLI option/flag, e.g., "--runtime kaggle".

    Returns
    -------
    str
        The resolved runtime name to be used for Hydra overrides.
    """
    if cli_value and str(cli_value).strip():
        runtime = str(cli_value).strip()
    else:
        runtime = os.environ.get(_ENV_RUNTIME, "").strip() or "default"
    # Normalize to lowercase for consistency
    return runtime.lower()


def set_runtime_env(runtime: str) -> None:
    """
    Persist the runtime selection to the process environment so that nested CLIs
    and subprocesses see the same choice without re-specifying it.
    """
    os.environ[_ENV_RUNTIME] = runtime


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _project_root() -> pathlib.Path:
    """
    Attempt to discover the repo root by walking up until we find a 'configs' directory.
    Fallback to CWD if not found.
    """
    p = pathlib.Path.cwd()
    for _ in range(10):
        if (p / "configs").exists():
            return p
        p = p.parent
    return pathlib.Path.cwd()


def compose_with_runtime(runtime: str, overrides: Optional[Iterable[str]] = None) -> DictConfig:
    """
    Compose a Hydra config with the given runtime override, plus any additional overrides.

    This function initializes Hydra in a context-managed way so repeated calls from the same
    process are safe. It uses the project's `configs/` directory as the search path.

    Parameters
    ----------
    runtime : str
        The runtime choice (e.g., "default", "local", "kaggle", "hpc", "docker", "ci").
    overrides : Optional[Iterable[str]]
        Any additional Hydra overrides to include in the composition.

    Returns
    -------
    DictConfig
        A composed Hydra config with `runtime=<name>` applied.
    """
    # Lazy import to avoid global Hydra initialization on module import.
    from hydra import initialize_config_dir, compose

    root = _project_root()
    config_dir = str(root / "configs")

    all_overrides: List[str] = [f"runtime={runtime}"]
    if overrides:
        all_overrides.extend(list(overrides))

    # Initialize Hydra with explicit config_dir to avoid ambiguity in packaged installs.
    with initialize_config_dir(config_dir=config_dir, job_name=f"compose:{runtime}"):
        cfg = compose(config_name="config", overrides=all_overrides)

    return cfg


def log_runtime_choice(runtime: str, cfg: Optional[DictConfig], extra_note: Optional[str] = None) -> None:
    """
    Append a minimal, structured line to v50_debug_log.md so we always have an auditable
    record of how the CLI was executed with regard to runtime choice.

    Format (single line, Markdown table friendly):
        2025-08-15T22:59:00Z | runtime=kaggle | config_root=/abs/path/configs | note=...

    This function is intentionally lightweight and never raises; any exceptions are swallowed
    so it does not interfere with the primary CLI action.
    """
    try:
        root = _project_root()
        log_path = root / "v50_debug_log.md"
        config_root = str((_project_root() / "configs").resolve())

        line = f"{_now_iso()} | runtime={runtime} | config_root={config_root}"
        if extra_note:
            line += f" | note={extra_note}"

        line += "\n"

        # Ensure file exists with a header on first write.
        if not log_path.exists():
            header = (
                "# SpectraMind V50 — Debug Log\n"
                "| timestamp (UTC) | runtime | config_root | note |\n"
                "|---|---|---|---|\n"
            )
            with log_path.open("w", encoding="utf-8") as f:
                f.write(header)

        with log_path.open("a", encoding="utf-8") as f:
            # Convert the simple line to a Markdown row for readability.
            parts = [p.strip() for p in line.strip().split("|")]
            # Expecting: ["2025-...", " runtime=...", " config_root=...", " note=..."]
            # Normalize to cells without keys for compactness, but keep values visible.
            ts = parts[0]
            rv = parts[1].split("=", 1)[1] if len(parts) > 1 and "=" in parts[1] else ""
            cr = parts[2].split("=", 1)[1] if len(parts) > 2 and "=" in parts[2] else ""
            nt = parts[3].split("=", 1)[1] if len(parts) > 3 and "=" in parts[3] else ""
            f.write(f"| {ts} | {rv} | {cr} | {nt} |\n")
    except Exception:
        # Never break the CLI; logging is best-effort.
        pass


def pretty_print_cfg(cfg: DictConfig) -> None:
    """
    Convenience function to print the composed configuration to stdout
    in a deterministic way for quick debugging or verification.
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))
