import os
import glob
import pytest
from omegaconf import OmegaConf

CONFIG_PATH = "src/spectramind/config"

def get_all_yaml_configs():
    return glob.glob(os.path.join(CONFIG_PATH, "**", "*.yaml"), recursive=True)

@pytest.mark.parametrize("config_file", get_all_yaml_configs())
def test_config_loads_without_error(config_file):
    """Ensure every Hydra YAML config loads correctly."""
    try:
        cfg = OmegaConf.load(config_file)
        assert cfg is not None
    except Exception as e:
        pytest.fail(f"Config {config_file} failed to load: {e}")

def test_at_least_one_config_exists():
    """Check that the config directory is not empty."""
    files = get_all_yaml_configs()
    assert len(files) > 0, "No Hydra config files found in src/spectramind/config/"
