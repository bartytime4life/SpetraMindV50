from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from .base import BaseSchema
from .types import NonEmptyStr, PlanetId
from .schema_registry import register_model


@register_model
class SymbolicRuleHit(BaseSchema):
    """A single symbolic rule's violation score and affected region summary."""

    rule_id: NonEmptyStr = Field(description="Stable rule identifier (e.g., 'nonnegativity', 'H2O_mask').")
    description: Optional[str] = Field(default=None, description="Human-readable description of the rule.")
    score: float = Field(ge=0.0, description="Violation magnitude or weighted score (higher = worse).")
    weight: float = Field(default=1.0, ge=0.0, description="Weight used in composite scoring.")
    affected_bins: List[int] = Field(default_factory=list, description="Indices of bins contributing most to the score.")


@register_model
class SymbolicViolationSummary(BaseSchema):
    """Top-K symbolic violations per planet and overall violation norms."""

    planet: PlanetId = Field(description="Planet this summary pertains to.")
    top_rules: List[SymbolicRuleHit] = Field(default_factory=list, description="Sorted by severity descending.")
    total_violation_norm: float = Field(ge=0.0, description="Aggregate norm across all rules.")
