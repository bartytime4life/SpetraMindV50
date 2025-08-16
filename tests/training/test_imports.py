"""
Basic import tests ensure the training package and all modules are importable.

These tests do not execute the Hydra-decorated CLI entrypoints; they only
verify that module-level imports succeed and that key symbols exist.
"""

import importlib

import pytest


def test_package_importable():
    pkg = importlib.import_module("src.spectramind.training")
    assert hasattr(pkg, "__doc__")


def test_train_v50_importable():
    pytest.importorskip("torch")
    mod = importlib.import_module("src.spectramind.training.train_v50")
    assert hasattr(mod, "main")


def test_predict_v50_importable():
    mod = importlib.import_module("src.spectramind.training.predict_v50")
    assert hasattr(mod, "main")


def test_train_mae_v50_importable():
    mod = importlib.import_module("src.spectramind.training.train_mae_v50")
    assert hasattr(mod, "main")


def test_train_contrastive_v50_importable():
    mod = importlib.import_module("src.spectramind.training.train_contrastive_v50")
    assert hasattr(mod, "main")


def test_tune_temperature_v50_importable():
    mod = importlib.import_module("src.spectramind.training.tune_temperature_v50")
    assert hasattr(mod, "main")


def test_train_corel_importable():
    mod = importlib.import_module("src.spectramind.training.train_corel")
    assert hasattr(mod, "main")


def test_run_ablation_trials_importable():
    mod = importlib.import_module("src.spectramind.training.run_ablation_trials")
    assert hasattr(mod, "main")


def test_utils_importable_and_symbols():
    mod = importlib.import_module("src.spectramind.training.utils")
    for name in [
        "train_from_config",
        "predict_from_config",
        "train_mae_from_config",
        "train_contrastive_from_config",
        "tune_temperature_from_config",
        "train_corel_from_config",
        "run_ablation_trials_from_config",
    ]:
        assert hasattr(mod, name), f"utils missing {name}"
