from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from .base import BaseSchema
from .types import PlanetId
from .schema_registry import register_model


@register_model
class BinStat(BaseSchema):
    """Per-bin diagnostic metrics for μ/σ evaluation and error localization."""

    bin_index: int = Field(ge=0, description="Spectral bin index.")
    gll: float = Field(description="Gaussian log-likelihood contribution for the bin.")
    rmse: Optional[float] = Field(default=None, ge=0.0, description="Root mean square error (optional).")
    mae: Optional[float] = Field(default=None, ge=0.0, description="Mean absolute error (optional).")
    entropy: Optional[float] = Field(default=None, ge=0.0, description="Entropy proxy or spectral entropy.")
    calibrated_sigma: Optional[float] = Field(default=None, ge=0.0, description="Post-calibration σ for the bin.")


@register_model
class GlobalMetricSummary(BaseSchema):
    """Global metrics aggregated over bins/planets."""

    num_planets: int = Field(ge=0, description="Number of planets included.")
    num_bins: int = Field(ge=0, description="Number of spectral bins per planet.")
    mean_gll: float = Field(description="Mean GLL across all bins/planets.")
    median_gll: Optional[float] = Field(default=None, description="Median GLL if computed.")
    mean_rmse: Optional[float] = Field(default=None, ge=0.0, description="Mean RMSE if available.")
    mean_mae: Optional[float] = Field(default=None, ge=0.0, description="Mean MAE if available.")


@register_model
class DiagnosticSummary(BaseSchema):
    """Complete diagnostics bundle suitable for dashboard embedding and manifest assembly."""

    planets: List[PlanetId] = Field(default_factory=list, description="Order of planets covered in this summary.")
    global_metrics: GlobalMetricSummary = Field(description="Global metric summary.")
    per_bin: List[BinStat] = Field(default_factory=list, description="Flattened per-bin stats over the dataset.")
    notes: Optional[str] = Field(default=None, description="Optional text notes or provenance.")
