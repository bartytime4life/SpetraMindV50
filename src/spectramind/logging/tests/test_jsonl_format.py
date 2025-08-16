"""
Minimal JSONL format test to ensure we serialize a dict to a single line and parse it back.
"""
import json
import pytest

@pytest.mark.unit
def test_jsonl_single_line_roundtrip():
    entry = {"timestamp": "2025-08-16T00:00:00Z", "level": "INFO", "msg": "test", "run_id": "test-run-0001"}
    serialized = json.dumps(entry, ensure_ascii=False)
    assert serialized.count("\n") == 0
    parsed = json.loads(serialized)
    assert parsed == entry
