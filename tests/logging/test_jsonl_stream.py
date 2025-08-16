import json
from spectramind.logging.jsonl_handler import JSONLHandler


def test_jsonl_logging(tmp_path):
    """Validate JSONL logging writes structured events."""
    log_file = tmp_path / "events.jsonl"
    handler = JSONLHandler(filename=log_file)
    record = {"event": "unit_test", "status": "ok"}
    handler.emit(record)

    assert log_file.exists()
    with open(log_file) as f:
        line = f.readline()
        data = json.loads(line)
    assert data["event"] == "unit_test"
    assert data["status"] == "ok"
