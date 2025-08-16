from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spectramind.conf_helpers import cli_override_parser


def test_parser_types():
    p = cli_override_parser(
        ["train.lr=0.01", "flags.debug=True", "model=fgs1_mamba", "opt.list=[1,2,'x']"]
    )
    assert p["train.lr"] == 0.01
    assert p["flags.debug"] is True
    assert p["model"] == "fgs1_mamba"
    assert p["opt.list"] == [1, 2, "x"]
