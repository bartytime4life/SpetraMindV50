"""
Pytest configuration for SpectraMind V50 training tests.

- Sets logging to INFO to capture training utils logs.
- Provides a lightweight OmegaConf DictConfig fixture with stable defaults.
- Guards Hydra's global state from leaking across tests by avoiding hydra.main entrypoints.
"""

import logging
import os
import sys

import pytest
from omegaconf import OmegaConf


def pytest_configure(config):
    # Ensure repository root (containing src/) is on sys.path so imports work in CI and local runs.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


@pytest.fixture(autouse=True)
def _set_logging_level(caplog):
    caplog.set_level(logging.INFO)
    yield


@pytest.fixture
def cfg_minimal():
    """
    Returns a minimal DictConfig for unit tests that call training utils directly
    without invoking full Hydra or heavy pipelines.
    """
    base = {
        "hash": "TEST-HASH-UNIT",
        "runtime": {"seed": 123, "device": "cpu"},
        "paths": {"artifacts": "artifacts/test", "logs": "logs/test"},
        "training": {"epochs": 1, "batch_size": 2, "amp": False},
        "ablation": {"enabled": False, "mutations": []},
        "corel": {"enabled": False, "model": {"hidden": 8, "layers": 1}},
    }
    return OmegaConf.create(base)
