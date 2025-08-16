from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from .base import BaseSchema
from .schema_registry import register_model


@register_model
class BinCalibration(BaseSchema):
    """Per-bin calibration quality."""

    bin_index: int = Field(ge=0, description="Spectral bin index.")
    nominal_sigma: Optional[float] = Field(default=None, ge=0.0, description="Pre-calibration σ.")
    calibrated_sigma: Optional[float] = Field(default=None, ge=0.0, description="Post-calibration σ.")
    residual_abs: Optional[float] = Field(default=None, ge=0.0, description="|μ - y| residual for the bin.")
    zscore: Optional[float] = Field(default=None, description="z-score under calibrated σ.")
    covered: Optional[bool] = Field(default=None, description="Whether residual is within target quantile envelope.")


@register_model
class CalibrationSummary(BaseSchema):
    """Coverage and calibration summary across all bins."""

    target_coverage: float = Field(ge=0.0, le=1.0, description="Desired probability coverage (e.g., 0.9).")
    observed_coverage: float = Field(ge=0.0, le=1.0, description="Empirical coverage achieved.")
    num_bins: int = Field(ge=0, description="Total bins evaluated.")
    per_bin: List[BinCalibration] = Field(default_factory=list, description="Per-bin calibration records.")
