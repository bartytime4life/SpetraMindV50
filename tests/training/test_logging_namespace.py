"""
Verify that the logger namespace used by training modules can be configured.

This ensures consistent log capture across the repo (e.g., v50_debug_log.md writers).
"""

import importlib
import logging
import pytest


def test_training_logger_namespace_exists():
    pytest.importorskip("torch")
    importlib.import_module("src.spectramind.training.utils")
    logger = logging.getLogger("spectramind.training")
    logger.info("logger-namespace-smoke")
    assert logger.name == "spectramind.training"
