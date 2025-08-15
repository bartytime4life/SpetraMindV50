# SPDX-License-Identifier: MIT

"""Pydantic schema definitions for SpectraMind configurations."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class V50ConfigSchema(BaseModel):
    """Canonical configuration schema for v50."""

    # Core run metadata
    experiment_name: str = Field(..., description="Human-readable experiment label")
    seed: int = Field(..., description="Global RNG seed")
    device: str = Field(..., description="Compute target, e.g., 'cuda:0' or 'cpu'")
    data_dir: str = Field(..., description="Path to dataset root")
    output_dir: str = Field(..., description="Path to write artifacts and checkpoints")

    # Encoders/Decoders
    encoder: Dict[str, Any] = Field(..., description="Encoder configuration (FGS1/AIRS)")
    decoder: Dict[str, Any] = Field(..., description="Decoder configuration (mu/sigma heads)")

    # Training block
    training: Dict[str, Any] = Field(..., description="Optimizer, schedulers, epochs, etc.")

    # Optional blocks
    calibration: Optional[Dict[str, Any]] = Field(None, description="Uncertainty calibration settings")
    symbolic: Optional[Dict[str, Any]] = Field(None, description="Symbolic rules/weights/regions")


def get_json_schema() -> Dict[str, Any]:
    """Export the JSON Schema for the configuration."""
    return V50ConfigSchema.schema()
