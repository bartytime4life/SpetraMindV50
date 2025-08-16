# tests/logging/test_hydra_logging_group.py
# Validates that logging group variants load and basic keys exist.

import os
from omegaconf import OmegaConf

def _load(rel):
    return OmegaConf.load(os.path.join("configs", rel))

def test_logging_base_loads():
    cfg = _load("logging/base.yaml")
    assert "logging" in cfg and "handlers" in cfg.logging

def test_logging_console_loads():
    cfg = _load("logging/console.yaml")
    assert "logging" in cfg and "handlers" in cfg.logging and "console" in cfg.logging.handlers

def test_logging_file_loads():
    cfg = _load("logging/file.yaml")
    assert cfg.logging.handlers.file.filename.endswith("v50_debug_log.md")

def test_logging_jsonl_loads():
    cfg = _load("logging/jsonl.yaml")
    assert cfg.logging.handlers.jsonl.filename.endswith("v50_event_log.jsonl")

def test_logging_mlflow_loads():
    cfg = _load("logging/mlflow.yaml")
    assert "mlflow" in cfg
