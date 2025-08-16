# Basic template sanity tests; functional tests live under tests/logging/.
import os
from omegaconf import OmegaConf

BASE = "src/spectramind/logging/hydra_templates"

def test_base_template_loads():
    cfg = OmegaConf.load(os.path.join(BASE, "base.yaml"))
    assert "logging" in cfg

def test_console_template_loads():
    cfg = OmegaConf.load(os.path.join(BASE, "logging_console.yaml"))
    assert "logging" in cfg

def test_file_template_loads():
    cfg = OmegaConf.load(os.path.join(BASE, "logging_file.yaml"))
    assert cfg.logging.handlers.file.filename.endswith("v50_debug_log.md")

def test_jsonl_template_loads():
    cfg = OmegaConf.load(os.path.join(BASE, "logging_jsonl.yaml"))
    assert cfg.logging.handlers.jsonl.formatter == "json"

def test_mlflow_template_loads():
    cfg = OmegaConf.load(os.path.join(BASE, "logging_mlflow.yaml"))
    assert "mlflow" in cfg
