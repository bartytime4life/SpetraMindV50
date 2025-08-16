from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from omegaconf import OmegaConf

from spectramind.conf_helpers import load_yaml, save_config


def test_save_and_load(tmp_path):
    cfg = {"train": {"lr": 0.001}, "model": "fgs1_mamba"}
    out = tmp_path / "cfg.yaml"
    save_config(OmegaConf.create(cfg), str(out))
    assert out.exists()
    loaded = load_yaml(str(out))
    assert float(loaded.train.lr) == 0.001
    assert str(loaded.model) == "fgs1_mamba"
