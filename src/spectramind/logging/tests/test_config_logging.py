"""
Test Hydra logging configuration loading.
"""
import yaml
from pathlib import Path

def test_hydra_logging_config_loads():
    cfg_path = Path("configs/logging/default.yaml")
    assert cfg_path.exists(), "Hydra logging config missing."
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert "handlers" in cfg, "Config must define handlers."
    assert "formatters" in cfg, "Config must define formatters."
