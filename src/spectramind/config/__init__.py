"""
SpectraMind V50 - Config Package

This package contains Hydra-compatible configuration files and utilities
for reproducible, mission-ready training, inference, calibration, and diagnostics.

All configs follow NASA-grade documentation rigor and integrate with schema validation,
CLI orchestration, and logging. Use `hydra.compose` or CLI `--config-path` to load.
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent

__all__ = ["CONFIG_DIR"]
