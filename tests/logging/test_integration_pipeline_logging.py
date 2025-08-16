import logging
from spectramind.logging import LoggingConfig, init_logging, get_logger


def test_integration_logging_pipeline(tmp_path, capsys):
    """Test that full pipeline logging writes to console, file, and JSONL."""
    log_dir = tmp_path / "logs"
    cfg = LoggingConfig(log_dir=str(log_dir), console=True, file=True, jsonl=True)
    init_logging(cfg)

    logger = get_logger("spectramind.pipeline")
    logger.info("Integration log message")

    captured = capsys.readouterr()
    assert "Integration log message" in captured.out

    logfile = log_dir / "spectramind.log"
    jsonl = log_dir / "events.jsonl"
    assert logfile.exists()
    assert jsonl.exists()
    assert "Integration log message" in logfile.read_text()
