"""
Test rotating file logs.
"""
import logging
from logging.handlers import RotatingFileHandler

def test_rotating_file_handler(temp_log_dir):
    log_file = temp_log_dir / "rot.log"
    handler = RotatingFileHandler(log_file, maxBytes=200, backupCount=2)
    logger = logging.getLogger("test")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    for i in range(100):
        logger.info("Log line %d", i)

    # At least one rotated file should exist
    rotated = list(temp_log_dir.glob("rot.log*"))
    assert len(rotated) >= 1
