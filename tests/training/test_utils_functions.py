"""
Unit tests for the lightweight training utils "from_config" functions.

We assert that:
- Each function accepts a DictConfig-like object.
- Each function emits an INFO log line with the expected prefix.
These tests avoid heavy GPU/IO work and serve as interface/contract checks.
"""

import importlib
import re
from typing import Callable

import pytest

Utils = importlib.import_module("src.spectramind.training.utils")


def _assert_logged(caplog, pattern: str):
    joined = "\n".join([f"{r.levelname}:{r.name}:{r.message}" for r in caplog.records])
    assert re.search(
        pattern, joined
    ), f"Expected log pattern not found: {pattern}\nLogs:\n{joined}"


@pytest.mark.parametrize(
    "fn_name,expected",
    [
        (
            "train_from_config",
            r"\[train_from_config\] Starting training with config hash=TEST-HASH-UNIT",
        ),
        (
            "predict_from_config",
            r"\[predict_from_config\] Running prediction with config hash=TEST-HASH-UNIT",
        ),
        (
            "train_mae_from_config",
            r"\[train_mae_from_config\] Running MAE pretraining\.\.\.",
        ),
        (
            "train_contrastive_from_config",
            r"\[train_contrastive_from_config\] Running contrastive pretraining\.\.\.",
        ),
        (
            "tune_temperature_from_config",
            r"\[tune_temperature_from_config\] Running temperature scaling\.\.\.",
        ),
        (
            "train_corel_from_config",
            r"\[train_corel_from_config\] Training Spectral COREL GNN\.\.\.",
        ),
        (
            "run_ablation_trials_from_config",
            r"\[run_ablation_trials_from_config\] Running ablation studies\.\.\.",
        ),
    ],
)
def test_utils_log_messages(cfg_minimal, caplog, fn_name, expected):
    fn: Callable = getattr(Utils, fn_name)
    fn(cfg_minimal)
    _assert_logged(caplog, expected)
