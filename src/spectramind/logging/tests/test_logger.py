import logging
from spectramind.logging import init_logging, get_logger, LoggingConfig


def test_logger_info(tmp_path):
    cfg = LoggingConfig(log_dir=str(tmp_path))
    init_logging(cfg)
    log = get_logger("test")
    log.info("Hello World")
    assert True  # reached without error
