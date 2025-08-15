"""
SpectraMind V50 — Reporting Subsystem
-------------------------------------

This package provides a complete, reproducible reporting pipeline:

* Data collection from diagnostics outputs
* Schema validation (YAML/JSON Schema)
* Interactive charts (Plotly) with Matplotlib fallback
* Markdown/HTML (and optional PDF) export
* CLI integration (`spectramind report ...`)
* Console + rotating-file logs + JSONL event stream
* Optional MLflow/W&B sync hooks
* Git & ENV capture baked into report metadata
* Hydra-safe logging defaults
"""

from .report_generator import ReportGenerator, ReportConfig
from .report_data_collector import ReportDataCollector, CollectorConfig
from .export_manager import ExportManager, ExportConfig

__all__ = [
    "ReportGenerator",
    "ReportConfig",
    "ReportDataCollector",
    "CollectorConfig",
    "ExportManager",
    "ExportConfig",
]
