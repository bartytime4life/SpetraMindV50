import pytest
from hydra import compose, initialize

from src.spectramind.config import CONFIG_DIR
from src.spectramind.config.validator import validate_config
from src.spectramind.config.registry import load_config

CONFIG_NAMES = ["model", "training", "calibration", "diagnostics", "logging"]


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_validate_yaml(name: str):
    cfg = validate_config(CONFIG_DIR / f"{name}.yaml")
    assert cfg is not None


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_registry_loads(name: str):
    cfg = load_config(name)
    assert cfg is not None


def test_defaults_compose():
    with initialize(version_base=None, config_path=".."):
        cfg = compose(config_name="defaults")
    assert "fgs1_encoder" in cfg
    assert "optimizer" in cfg
    assert "temperature_scaling" in cfg
    assert "fft" in cfg
    assert "console" in cfg
