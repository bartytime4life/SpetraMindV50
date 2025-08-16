"""
Hydra/structured logging configuration tests.

These tests validate that a logging config file is present and structurally sane.
They DO NOT enforce exact handler names to avoid breaking local variants.
"""
from pathlib import Path
import yaml
import pytest

CANDIDATE_PATHS = [
    Path("configs/logging/default.yaml"),
    Path("conf/logging/default.yaml"),
    Path("configs/logging.yaml"),
]

@pytest.mark.smoke
def test_logging_config_file_exists():
    exists = [p for p in CANDIDATE_PATHS if p.exists()]
    assert exists, f"No logging config found in any of: {', '.join(str(p) for p in CANDIDATE_PATHS)}"

@pytest.mark.smoke
def test_logging_config_minimal_keys():
    exists = [p for p in CANDIDATE_PATHS if p.exists()]
    cfg_path = exists[0]
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Accept both dictConfig or logging.config.dictConfig style structured configs
    # Check for some common top-level keys
    keys = set(cfg.keys())
    assert any(k in keys for k in ("handlers", "logging", "root", "formatters")), \
        f"Config {cfg_path} missing expected logging sections; found keys: {sorted(keys)}"
