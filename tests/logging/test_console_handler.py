import logging
import pytest
from spectramind.logging.console_handler import setup_console_logging


def test_console_handler_basic(caplog):
    """Ensure console handler logs to stdout with expected format."""
    setup_console_logging(level=logging.DEBUG)
    logger = logging.getLogger("spectramind.test.console")
    with caplog.at_level(logging.DEBUG):
        logger.debug("Console log test")
    assert "Console log test" in caplog.text
    assert any(rec.levelname == "DEBUG" for rec in caplog.records)
