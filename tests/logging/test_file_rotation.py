import logging
from spectramind.logging.file_handler import setup_rotating_file_logging


def test_rotating_file_logging(tmp_path):
    """Check rotating file logging creates a log file and rolls over."""
    log_file = tmp_path / "test.log"
    setup_rotating_file_logging(log_file, max_bytes=200, backup_count=2)

    logger = logging.getLogger("spectramind.test.file")
    for i in range(50):
        logger.info("This is line %d", i)

    assert log_file.exists()
    rotated_files = list(tmp_path.glob("test.log.*"))
    assert len(rotated_files) >= 1
