"""
Integration tests for Hydra and CLI config helpers in SpectraMind V50.

These tests validate that Hydra initialization works correctly
and that configs interpolate properly across groups.
"""

import sys
from pathlib import Path

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@pytest.mark.parametrize("overrides", [["+model.type=airs_gnn"], ["+train.epochs=2"]])
def test_hydra_integration(overrides):
    """Hydra config composition with overrides works."""
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="config_v50", overrides=overrides)
        assert isinstance(cfg, DictConfig)
        for override in overrides:
            key, val = override.split("=")
            clean_key = key.lstrip("+")
            node = cfg
            for sub in clean_key.split("."):
                node = node[sub]
            assert str(node) == val
