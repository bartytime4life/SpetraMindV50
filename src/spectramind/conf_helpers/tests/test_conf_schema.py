"""
Schema validation tests for SpectraMind V50 configs.

Ensures that required fields exist and symbolic-aware defaults
are enforced.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spectramind.conf_helpers.schema import V50ConfigSchema


def test_required_fields_exist():
    """Ensure schema-required fields are present in the Pydantic model."""
    fields = V50ConfigSchema.__fields__
    assert "experiment_name" in fields
    assert "seed" in fields
    assert "training" in fields


def test_check_required_fields_passes():
    """Ensure a valid config passes schema check."""
    cfg = {
        "experiment_name": "demo",
        "seed": 3,
        "device": "cpu",
        "data_dir": "./data",
        "output_dir": "./out",
        "encoder": {},
        "decoder": {},
        "training": {"epochs": 3},
    }
    model = V50ConfigSchema(**cfg)
    assert model.seed == 3


def test_check_required_fields_fails():
    """Ensure missing fields trigger schema error."""
    cfg = {"experiment_name": "demo"}
    with pytest.raises(ValidationError):
        V50ConfigSchema(**cfg)
