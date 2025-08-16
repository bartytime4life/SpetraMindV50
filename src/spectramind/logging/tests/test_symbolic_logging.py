"""
Symbolic loss logging tests.

We simulate a symbolic loss JSONL record and verify required keys exist.
"""
import json
from pathlib import Path
import pytest

REQUIRED = {"rule", "loss"}

@pytest.mark.unit
def test_symbolic_loss_logged(temp_log_dir):
    log_path = temp_log_dir / "symbolic_log.jsonl"
    record = {"rule": "smoothness", "loss": 0.012, "planet_id": "PL_0001", "bin": 42}
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    line = log_path.read_text(encoding="utf-8").splitlines()[0]
    parsed = json.loads(line)
    assert REQUIRED.issubset(parsed), f"Missing fields in symbolic record: {REQUIRED - set(parsed)}"
    assert isinstance(parsed["loss"], (int, float))
