import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from omegaconf import OmegaConf

from spectramind.conf_helpers import apply_overrides, cli_override_parser


def test_cli_overrides():
    base = OmegaConf.create({"a": {"b": 1}})
    overrides = cli_override_parser(["a.b=2", "c=3"])
    merged = apply_overrides(base, overrides)
    assert merged.a.b == 2
    assert merged.c == 3
