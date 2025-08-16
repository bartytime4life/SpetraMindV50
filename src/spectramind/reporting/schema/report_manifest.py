from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field

from .base import BaseSchema
from .types import HashStr, NonEmptyStr, PathLikeStr
from .diagnostic_summary import DiagnosticSummary
from .symbolic_violation import SymbolicViolationSummary
from .calibration import CalibrationSummary
from .html_report import HTMLReportManifest
from .schema_registry import register_model


@register_model
class DatasetMeta(BaseSchema):
    """Dataset-level provenance for the report package."""

    dataset_name: NonEmptyStr = Field(description="Identifier for the dataset (e.g., 'ariel-neurips-2025').")
    split: NonEmptyStr = Field(description="Data split (e.g., 'train', 'val', 'test', 'holdout').")
    num_planets: int = Field(ge=0, description="Count of planets included.")
    num_bins: int = Field(ge=0, description="Bins per planet.")


@register_model
class GeneratorMeta(BaseSchema):
    """Software provenance for the artifacts included in the report."""

    cli: NonEmptyStr = Field(description="CLI tool that produced the package (e.g., 'spectramind').")
    version: NonEmptyStr = Field(description="CLI/app version string.")
    config_hash: Optional[HashStr] = Field(default=None, description="Config hash used for generation.")
    artifacts_dir: Optional[PathLikeStr] = Field(default=None, description="Root directory for artifacts.")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment snapshot (subset).")


@register_model
class Artifact(BaseSchema):
    """A single named artifact in the package."""

    name: NonEmptyStr = Field(description="Logical name, stable within the package.")
    path: PathLikeStr = Field(description="Relative/absolute path to the artifact file.")
    kind: NonEmptyStr = Field(description="Type label (e.g., 'json', 'png', 'html', 'csv').")
    description: Optional[str] = Field(default=None, description="Optional human-readable details.")


@register_model
class ReportPackageManifest(BaseSchema):
    """Complete package manifest that downstream tools can validate, sign, or publish."""

    dataset: DatasetMeta = Field(description="Dataset/provenance metadata.")
    generator: GeneratorMeta = Field(description="Software provenance metadata.")
    diagnostics: Optional[DiagnosticSummary] = Field(default=None, description="Global/Per-bin diagnostics.")
    calibration: Optional[CalibrationSummary] = Field(default=None, description="Uncertainty calibration summary.")
    symbolic: Optional[List[SymbolicViolationSummary]] = Field(
        default=None, description="Per-planet symbolic summaries."
    )
    html_report: Optional[HTMLReportManifest] = Field(default=None, description="HTML UI manifest.")
    artifacts: List[Artifact] = Field(default_factory=list, description="All artifacts included in the package.")
