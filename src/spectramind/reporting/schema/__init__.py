from __future__ import annotations

from .base import BaseSchema
from .types import HashStr, PlanetId, NonEmptyStr, IsoDatetime, PathLikeStr
from .cli_log import CLILogEntry, CLILogBatch
from .diagnostic_summary import BinStat, DiagnosticSummary, GlobalMetricSummary
from .symbolic_violation import SymbolicRuleHit, SymbolicViolationSummary
from .calibration import BinCalibration, CalibrationSummary
from .html_report import HTMLAsset, HTMLReportManifest
from .report_manifest import (
    ReportPackageManifest,
    DatasetMeta,
    GeneratorMeta,
    Artifact,
)
from .schema_registry import (
    SchemaRegistry,
    registry,
    register_model,
    export_json_schemas,
    list_registered_models,
)

__all__ = [
    # Base/types
    "BaseSchema",
    "HashStr",
    "PlanetId",
    "NonEmptyStr",
    "IsoDatetime",
    "PathLikeStr",
    # CLI logs
    "CLILogEntry",
    "CLILogBatch",
    # Diagnostics
    "BinStat",
    "DiagnosticSummary",
    "GlobalMetricSummary",
    # Symbolic
    "SymbolicRuleHit",
    "SymbolicViolationSummary",
    # Calibration
    "BinCalibration",
    "CalibrationSummary",
    # HTML report assets/manifest
    "HTMLAsset",
    "HTMLReportManifest",
    # Package/report manifest
    "ReportPackageManifest",
    "DatasetMeta",
    "GeneratorMeta",
    "Artifact",
    # Registry
    "SchemaRegistry",
    "registry",
    "register_model",
    "export_json_schemas",
    "list_registered_models",
]
