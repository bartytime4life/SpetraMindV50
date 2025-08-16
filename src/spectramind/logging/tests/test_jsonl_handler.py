# SPDX-License-Identifier: MIT
import json

from spectramind.logging import LoggingConfig, get_logger, init_logging


def test_jsonl_schema(tmp_path):
    cfg = LoggingConfig(log_dir=str(tmp_path), jsonl=True, console=False, file=False)
    init_logging(cfg)
    log = get_logger("spectramind.jsonl")
    log.error("jsonl test error", extra={"planet_id": "V50-TEST"})
    with (tmp_path / "events.jsonl").open() as f:
        obj = json.loads(f.readline())
    assert "timestamp" in obj and "level" in obj and "logger" in obj and "message" in obj
    assert obj["logger"] == "spectramind.jsonl"
    assert obj["extras"]["planet_id"] == "V50-TEST"
