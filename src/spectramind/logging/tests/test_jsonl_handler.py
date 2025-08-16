from spectramind.logging import init_logging, get_logger, LoggingConfig
import json


def test_jsonl_handler(tmp_path):
    cfg = LoggingConfig(log_dir=str(tmp_path))
    init_logging(cfg)
    log = get_logger("jsonl")
    log.info("jsonl test")
    with open(tmp_path / "events.jsonl") as f:
        line = json.loads(f.readline())
        assert "message" in line
