import json
import logging
from spectramind.logging.jsonl_handler import JSONLHandler


def test_jsonl_logging(tmp_path):
    log_file = tmp_path / "events.jsonl"
    logger = logging.getLogger("jsonl_test")
    logger.handlers = []
    handler = JSONLHandler(log_file)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("unit test")

    assert log_file.exists()
    with open(log_file) as f:
        line = json.loads(f.readline())
    assert line["message"] == "unit test"
    assert line["level"] == "INFO"
