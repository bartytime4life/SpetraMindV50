import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from omegaconf import OmegaConf

from spectramind.conf_helpers import load_config, save_config


def test_load_and_save(tmp_path):
    cfg = OmegaConf.create({"a": 1})
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.a == 1
