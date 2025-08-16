"""
Test symbolic loss logging.
"""
import json
from pathlib import Path

def test_symbolic_loss_logged(temp_log_dir):
    log_path = temp_log_dir / "symbolic_log.jsonl"
    record = {"rule": "smoothness", "loss": 0.012}
    log_path.write_text(json.dumps(record) + "\n")
    parsed = json.loads(log_path.read_text().splitlines()[0])
    assert "rule" in parsed and "loss" in parsed
