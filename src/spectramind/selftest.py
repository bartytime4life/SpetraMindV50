from __future__ import annotations

"""
SpectraMind V50 — Self-test runner with runtime wiring.

This module composes a Hydra config with the selected runtime override and performs
a set of lightweight checks to verify that critical files, config groups, and CLI entries
are present and importable. It is intentionally fast and safe to run in CI.

Return codes:
0: success
1: failure detected
"""

import importlib
import sys
from typing import Optional, List

from omegaconf import DictConfig

from spectramind.conf_helpers.runtime import (
    determine_runtime,
    set_runtime_env,
    compose_with_runtime,
    log_runtime_choice,
)


def _check_import(module: str, errors: List[str]) -> None:
    try:
        importlib.import_module(module)
    except Exception as e:
        errors.append(f"Import failed: {module} -> {e}")


def main(cli_runtime: Optional[str] = None) -> int:
    """
    Entry point callable by the root CLI or programmatically.
    """
    runtime = determine_runtime(cli_runtime)
    set_runtime_env(runtime)
    cfg = compose_with_runtime(runtime, overrides=[])
    log_runtime_choice(runtime, cfg, extra_note="selftest.main")

    errors: List[str] = []

    # Minimal presence checks for frequently used modules. Extend as needed.
    _check_import("spectramind.cli.cli_core_v50", errors)
    _check_import("spectramind.cli.cli_submit", errors)
    _check_import("spectramind.cli.cli_diagnose", errors)
    _check_import("spectramind.conf_helpers.runtime", errors)

    # Sanity check that the composed config includes a 'runtime' group resolution.
    try:
        resolved = cfg.get("runtime", None)
        if not resolved or not isinstance(resolved, DictConfig):
            errors.append("Composed config missing 'runtime' group resolution.")
        else:
            name = resolved.get("name")
            if not name:
                errors.append("Composed runtime config lacks 'name' field.")
    except Exception as e:
        errors.append(f"Failed to access runtime config in composed cfg: {e}")

    if errors:
        for e in errors:
            print(f"[SELFTEST][FAIL] {e}")
        return 1

    print("[SELFTEST][OK] CLI + runtime wiring is healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main(None))

