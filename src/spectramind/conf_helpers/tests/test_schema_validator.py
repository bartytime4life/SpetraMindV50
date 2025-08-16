from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from omegaconf import OmegaConf

from spectramind.conf_helpers import validate_config


def test_validate(tmp_path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "type: object\nproperties:\n  train:\n    type: object\nrequired:\n  - train\n"
    )
    cfg = OmegaConf.create({"train": {"lr": 1e-3}})
    assert validate_config(cfg, str(schema_path)) is True
