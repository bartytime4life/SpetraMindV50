import pytest
from omegaconf import OmegaConf

CONFIG_FILE = "src/spectramind/config/config_v50.yaml"

def test_symbolic_loss_weights_valid():
    """Ensure symbolic loss weights are defined and non-negative."""
    cfg = OmegaConf.load(CONFIG_FILE)
    assert "symbolic" in cfg
    assert "loss_weights" in cfg.symbolic

    for rule, weight in cfg.symbolic.loss_weights.items():
        assert weight >= 0, f"Symbolic loss weight for {rule} is negative!"

def test_symbolic_profiles_linked():
    """Ensure symbolic profiles point to valid YAML files."""
    cfg = OmegaConf.load(CONFIG_FILE)
    if "profiles" in cfg.symbolic:
        for profile in cfg.symbolic.profiles.values():
            assert profile.endswith(".yaml"), f"Invalid profile reference: {profile}"
