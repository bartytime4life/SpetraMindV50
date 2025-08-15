"""SpectraMind V50 — Calibration Package

Exports and package metadata.
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

__all__ = [
    "Calibrator",
    "run_calibration_batch",
    "run_calibration_one",
]

try:
    version = _pkg_version("spectramind")
except Exception:  # pragma: no cover - package not installed
    version = "0.0.0-dev"

from .pipeline import Calibrator, run_calibration_batch, run_calibration_one  # noqa: E402,F401
