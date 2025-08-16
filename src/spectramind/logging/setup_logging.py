import logging
from pathlib import Path
from typing import Dict

from .console_handler import setup_console_logging
from .file_handler import setup_rotating_file_logging
from .jsonl_handler import JSONLHandler

class _JSONLLoggingHandler(logging.Handler):
    def __init__(self, filename: Path):
        super().__init__()
        self._jsonl = JSONLHandler(filename)

    def emit(self, record: logging.LogRecord) -> None:
        self._jsonl.emit({
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        })

def setup_logging(*, log_dir: Path, level: int = logging.INFO) -> Dict[str, Path]:
    """Configure console (file) + JSONL logging inside ``log_dir``."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers = []
    root.setLevel(level)

    console_file = log_dir / "console.log"
    setup_rotating_file_logging(console_file, max_bytes=1_000_000, backup_count=3, level=level, logger=root)

    jsonl_file = log_dir / "events.jsonl"
    root.addHandler(_JSONLLoggingHandler(jsonl_file))

    setup_console_logging(level=level, logger=root)
    return {"console": console_file, "jsonl": jsonl_file}
