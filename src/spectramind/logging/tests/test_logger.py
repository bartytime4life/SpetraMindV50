# SPDX-License-Identifier: MIT
import json
from pathlib import Path

from spectramind.logging import LoggingConfig, get_logger, init_logging


def test_logging_file_and_console(tmp_path, capsys):
    cfg = LoggingConfig(
        log_level="INFO",
        log_dir=str(tmp_path),
        jsonl=True,
        file=True,
        console=True,
        file_max_mb=1,
        file_backup_count=1,
    )
    init_logging(cfg)
    log = get_logger("spectramind.test")
    log.info("hello world", extra={"case": "file_and_console"})

    captured = capsys.readouterr()
    assert "hello world" in captured.out

    logfile = tmp_path / "spectramind.log"
    assert logfile.exists() and logfile.stat().st_size > 0

    jsonl_file = tmp_path / "events.jsonl"
    assert jsonl_file.exists()
    line = json.loads(jsonl_file.read_text().splitlines()[0])
    assert line["message"] == "hello world"
    assert line["level"] == "INFO"
