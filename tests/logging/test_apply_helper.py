# tests/logging/test_apply_helper.py
# Ensures apply.install_from_hydra installs dictConfig and creates log file when using file backend.

import os
from omegaconf import OmegaConf
from src.spectramind.logging.apply import install_from_hydra

BASE_CFG_PATH = "configs/logging/base.yaml"

def _load(name: str):
    base = OmegaConf.load(BASE_CFG_PATH)
    variant = OmegaConf.load(f"configs/logging/{name}.yaml")
    return OmegaConf.merge(base, variant)

def test_install_console(tmp_path, monkeypatch):
    cfg = _load("console")
    monkeypatch.chdir(tmp_path)
    install_from_hydra(cfg)


def test_install_file_creates_log(tmp_path, monkeypatch):
    cfg = _load("file")
    monkeypatch.chdir(tmp_path)
    install_from_hydra(cfg)
    assert os.path.exists(tmp_path / "logs" / "v50_debug_log.md")

def test_install_jsonl_creates_event_log(tmp_path, monkeypatch):
    cfg = _load("jsonl")
    monkeypatch.chdir(tmp_path)
    install_from_hydra(cfg)
    assert os.path.exists(tmp_path / "logs" / "v50_event_log.jsonl")
