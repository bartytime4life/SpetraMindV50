import os
import re
import json
import glob
import logging
from dataclasses import dataclass


@dataclass
class CollectorConfig:
    """Controls how we discover and merge downstream diagnostics into a unified view."""
    # Filenames we try to consume best-effort
    diagnostic_summary_name: str = "diagnostic_summary.json"
    fft_summary_name: str = "fft_summary.json"
    calibration_summary_name: str = "calibration_summary.json"
    projection_summary_name: str = "projection_summary.json"
    symbolic_summary_name: str = "symbolic_summary.json"
    cli_log_summary_name: str = "cli_log_summary.json"


class ReportDataCollector:
    """Aggregates diagnostics JSON (and optional CSV) into a single dict for reporting.
    All keys are optional — the report is resilient to missing inputs."""

    def __init__(self, cfg: CollectorConfig = CollectorConfig(), logger: logging.Logger = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    # -----------------
    # Helpers
    # -----------------
    def _load_json_if_exists(self, path: str):
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read JSON ({path}): {e}")
        return {}

    def _find_first(self, base_dir: str, name: str) -> str:
        """Find first matching file with given basename anywhere under base_dir."""
        for root, _, files in os.walk(base_dir):
            if name in files:
                return os.path.join(root, name)
        return ""

    # -----------------
    # Entry
    # -----------------
    def collect(self, diagnostics_dir: str) -> dict:
        """Collect all recognized summaries from diagnostics_dir (recursive)."""
        collected = {}

        # Diagnostic summary
        p = self._find_first(diagnostics_dir, self.cfg.diagnostic_summary_name)
        collected["diagnostic_summary"] = self._load_json_if_exists(p)

        # FFT summary
        p = self._find_first(diagnostics_dir, self.cfg.fft_summary_name)
        collected["fft_summary"] = self._load_json_if_exists(p)

        # Calibration summary
        p = self._find_first(diagnostics_dir, self.cfg.calibration_summary_name)
        collected["calibration_summary"] = self._load_json_if_exists(p)

        # Projections (UMAP/t-SNE)
        p = self._find_first(diagnostics_dir, self.cfg.projection_summary_name)
        collected["projection_summary"] = self._load_json_if_exists(p)

        # Symbolic
        p = self._find_first(diagnostics_dir, self.cfg.symbolic_summary_name)
        collected["symbolic_summary"] = self._load_json_if_exists(p)

        # CLI log summary
        p = self._find_first(diagnostics_dir, self.cfg.cli_log_summary_name)
        collected["cli_log_summary"] = self._load_json_if_exists(p)

        # Log what we found
        found = [k for k, v in collected.items() if v]
        missing = [k for k, v in collected.items() if not v]
        self.logger.info(f"Collected inputs: {found}. Missing: {missing}.")
        return collected
