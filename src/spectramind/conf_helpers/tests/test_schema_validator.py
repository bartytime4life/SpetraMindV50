import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spectramind.conf_helpers import load_config, validate_config


def test_validate_config(tmp_path):
    cfg_yaml = tmp_path / "cfg.yaml"
    schema_json = tmp_path / "schema.json"

    cfg_yaml.write_text("a: 1\n", encoding="utf-8")
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}},
        "required": ["a"],
    }
    schema_json.write_text(json.dumps(schema), encoding="utf-8")

    cfg = load_config(cfg_yaml)
    assert validate_config(cfg, schema_json)
