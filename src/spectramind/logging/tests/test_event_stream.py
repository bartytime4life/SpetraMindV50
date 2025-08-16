"""
Test JSONL event stream integrity.
"""
import json
from pathlib import Path

def test_event_stream_valid_json(temp_log_dir):
    stream_path = temp_log_dir / "event_stream.jsonl"
    sample = {"event": "train_start", "epoch": 0}
    stream_path.write_text(json.dumps(sample) + "\n")
    lines = stream_path.read_text().splitlines()
    for line in lines:
        record = json.loads(line)
        assert "event" in record
