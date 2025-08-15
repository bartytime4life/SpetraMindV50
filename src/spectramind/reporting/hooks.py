import os
import json
import logging
from typing import Optional

from .report_generator import ReportGenerator, ReportConfig
from .report_data_collector import CollectorConfig
from .export_manager import ExportConfig


def on_training_complete(
    diagnostics_dir: str = "artifacts/diagnostics",
    outputs_dir: str = "reports",
    version_suffix: Optional[str] = None,
    open_after: bool = False,
    enable_pdf: bool = False,
    logger: logging.Logger = None,
) -> dict:
    """Hook to be called by training pipeline. Generates an interim report."""
    logger = logger or logging.getLogger("spectramind.reporting.hooks")
    version = f"v_train_{version_suffix}" if version_suffix else "v_train"
    gen = ReportGenerator(
        report_cfg=ReportConfig(
            diagnostics_dir=diagnostics_dir,
            outputs_dir=outputs_dir,
            report_version=version,
            title="SpectraMind V50 — Training Report",
            subtitle="Pipeline Hook",
            open_after=open_after,
            enable_pdf=enable_pdf,
        ),
        collector_cfg=CollectorConfig(),
        export_cfg=ExportConfig(),
        logger=logger,
    )
    return gen.render()


def on_inference_complete(
    diagnostics_dir: str = "artifacts/diagnostics",
    outputs_dir: str = "reports",
    version_suffix: Optional[str] = None,
    open_after: bool = False,
    enable_pdf: bool = False,
    logger: logging.Logger = None,
) -> dict:
    """Hook to be called by inference/submission pipeline. Generates a final diagnostics report."""
    logger = logger or logging.getLogger("spectramind.reporting.hooks")
    version = f"v_infer_{version_suffix}" if version_suffix else "v_infer"
    gen = ReportGenerator(
        report_cfg=ReportConfig(
            diagnostics_dir=diagnostics_dir,
            outputs_dir=outputs_dir,
            report_version=version,
            title="SpectraMind V50 — Inference Diagnostics Report",
            subtitle="Pipeline Hook",
            open_after=open_after,
            enable_pdf=enable_pdf,
        ),
        collector_cfg=CollectorConfig(),
        export_cfg=ExportConfig(),
        logger=logger,
    )
    return gen.render()
