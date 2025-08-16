"""
Tests for reproducibility helpers: seeds, env capture, config hashes.
"""

import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spectramind.utils.reproducibility import set_all_seeds
from spectramind.conf_helpers.env_capture import log_environment
from spectramind.conf_helpers.hashing import config_hash


def test_seed_reproducibility():
    """Ensure setting seed makes random sequences reproducible."""
    set_all_seeds(42)
    first = [random.randint(0, 100) for _ in range(5)]
    set_all_seeds(42)
    second = [random.randint(0, 100) for _ in range(5)]
    assert first == second


def test_env_capture(tmp_path):
    """Ensure environment capture writes expected file."""
    path = tmp_path / "env.json"
    log_environment(str(path))
    assert os.path.exists(path)
    text = path.read_text()
    assert "python_version" in text


def test_config_hash_stability():
    """Ensure config hash is deterministic for same config."""
    cfg = {"a": 1, "b": {"c": 2}}
    h1 = config_hash(cfg)
    h2 = config_hash(cfg)
    assert h1 == h2
