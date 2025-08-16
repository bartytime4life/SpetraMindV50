"""
Rotating file handler behavior tests.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import pytest

@pytest.mark.unit
def test_rotating_file_handler(temp_log_dir):
    log_file = temp_log_dir / "rot.log"
    handler = RotatingFileHandler(log_file, maxBytes=512, backupCount=2)
    logger = logging.getLogger("spectramind.test.rotate")
    # Avoid test pollution across runs
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    for i in range(1000):
        logger.info("Log line %04d - rotating test payload", i)
    rotated = sorted(temp_log_dir.glob("rot.log*"))
    # Should have the base file + at least one rotated
    assert len(rotated) >= 2, f"Expected rotation files, found: {[p.name for p in rotated]}"
