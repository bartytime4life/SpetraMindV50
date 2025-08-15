"""
SpectraMind V50 Diagnostics Package
Mission-grade, neuro-symbolic, physics-informed diagnostic tools.

Features:

* Console + rotating file logs
* JSONL event stream
* Optional MLflow / W&B sync toggles (no-op if not configured)
* Git/ENV capture for reproducibility
* Hydra-safe config loading (YAML/JSON via OmegaConf if available; graceful fallback)
* Unified CLI via `python -m src.spectramind.diagnostics`

Submodules are imported lazily by the main CLI to keep import times minimal.
"""

__version__ = "0.1.0"
