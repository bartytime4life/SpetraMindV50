"""
JSONL event stream integrity tests.

Ensures each line is valid JSON with required minimal fields.
"""
import json
from pathlib import Path
import pytest
from datetime import datetime, timezone

REQUIRED_FIELDS = {"event", "ts"}

def _write_jsonl(sample_path: Path, rows):
    with sample_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

@pytest.mark.unit
def test_event_stream_valid_json_and_fields(temp_log_dir):
    stream_path = temp_log_dir / "event_stream.jsonl"
    rows = [
        {"event": "train_start", "ts": datetime.now(timezone.utc).isoformat(), "epoch": 0},
        {"event": "epoch_end", "ts": datetime.now(timezone.utc).isoformat(), "epoch": 0, "loss": 0.123},
    ]
    _write_jsonl(stream_path, rows)

    with stream_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            rec = json.loads(line)
            missing = REQUIRED_FIELDS - set(rec)
            assert not missing, f"Line {i} missing fields: {missing}"
            assert isinstance(rec["event"], str)
            assert isinstance(rec["ts"], str)
