import pytest
from omegaconf import OmegaConf, DictConfig

CONFIG_FILE = "src/spectramind/config/config_v50.yaml"

def test_main_config_exists():
    """Ensure the primary config_v50.yaml file exists."""
    cfg = OmegaConf.load(CONFIG_FILE)
    assert isinstance(cfg, DictConfig)

def test_required_sections_present():
    """Check essential config sections are defined."""
    cfg = OmegaConf.load(CONFIG_FILE)
    required = ["model", "train", "data", "symbolic", "diagnostics"]
    for r in required:
        assert r in cfg, f"Missing required config section: {r}"
