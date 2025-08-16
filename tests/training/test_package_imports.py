"""Basic import tests for the SpectraMind training package."""

import importlib
import pytest


def test_training_package_public_api():
    pytest.importorskip("torch")
    pkg = importlib.import_module("src.spectramind.training")
    for name in [
        "V50Trainer",
        "build_dataloaders",
        "build_optimizer",
        "build_scheduler",
        "compute_total_loss",
        "compute_metrics",
    ]:
        assert hasattr(pkg, name), f"training package missing {name}"
