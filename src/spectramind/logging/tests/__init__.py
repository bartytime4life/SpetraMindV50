"""
SpectraMind V50 - Logging test suite package initializer.

This package provides mission-ready pytest coverage for the logging subsystem:
- Hydra config loading & schema sanity checks
- JSONL event stream integrity
- Rotating file handlers
- CLI integration & version logging to v50_debug_log.md
- Symbolic loss logging metadata
- Reproducibility metadata (env, git hash) presence checks when available
"""
