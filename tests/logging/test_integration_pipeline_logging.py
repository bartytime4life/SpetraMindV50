import logging
from spectramind.logging.setup_logging import setup_logging


def test_integration_logging_pipeline(tmp_path):
    """Test that full pipeline logging writes to console, file, and JSONL."""
    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir, level=logging.INFO)

    logger = logging.getLogger("spectramind.pipeline")
    logger.info("Integration log message")

    console = log_dir / "console.log"
    jsonl = log_dir / "events.jsonl"
    assert console.exists()
    assert jsonl.exists()
    assert any("Integration log message" in open(console).read() for _ in [0])
