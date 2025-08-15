# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Inference Subsystem Package

This package provides mission-grade inference, ensemble aggregation, uncertainty
calibration, diagnostics, and packaging utilities for the NeurIPS 2025 Ariel Data
Challenge.

Exports a minimal surface for external CLIs to import without side effects.
"""

from .utils_infer import (
    InferenceConfig,
    ensure_run_dir,
    setup_logging_stack,
    compute_config_hash,
    write_json,
    read_json,
    write_jsonl_event,
    append_to_debug_log,
)

__all__ = [
    "InferenceConfig",
    "ensure_run_dir",
    "setup_logging_stack",
    "compute_config_hash",
    "write_json",
    "read_json",
    "write_jsonl_event",
    "append_to_debug_log",
]
