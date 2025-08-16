"""
Tests for configuration loader helpers in SpectraMind V50.

These tests validate that Hydra configs, schema enforcement,
and reproducibility helpers in ``spectramind.conf_helpers``
work as intended.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spectramind.conf_helpers import load_config
from spectramind.conf_helpers.config_loader import save_config
from spectramind.conf_helpers.validators import validate_config
from spectramind.conf_helpers.hashing import config_hash


@pytest.fixture
def dummy_config_dict():
    """Return a minimal dummy config for testing."""
    return {
        "experiment_name": "test_run",
        "seed": 123,
        "device": "cpu",
        "data_dir": "./data",
        "output_dir": "./out",
        "encoder": {"type": "fgs1_mamba", "hidden_dim": 64},
        "decoder": {"type": "basic"},
        "training": {"epochs": 1, "batch_size": 2},
    }


def test_load_and_validate_config(dummy_config_dict):
    """Ensure configs load and validate correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "dummy.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(dummy_config_dict, f)

        cfg = load_config(cfg_path)
        assert cfg.experiment_name == "test_run"
        validated = validate_config(cfg)
        assert validated.seed == 123


def test_save_config_snapshot(dummy_config_dict):
    """Ensure config snapshots are written correctly with hash consistency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = OmegaConf.create(dummy_config_dict)
        out_path = os.path.join(tmpdir, "snapshot.yaml")
        save_config(cfg, out_path)
        assert os.path.exists(out_path)
        with open(out_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["seed"] == 123
        assert config_hash(loaded) == config_hash(dummy_config_dict)
