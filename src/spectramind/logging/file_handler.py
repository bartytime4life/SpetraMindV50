import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

def setup_rotating_file_logging(filename: Path, *, max_bytes: int = 10 * 1024 * 1024,
                                backup_count: int = 5, level: int = logging.INFO,
                                logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Attach a rotating file handler writing to ``filename`` on the given logger."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = logger or logging.getLogger()
    target.setLevel(level)
    if not any(isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path for h in target.handlers):
        handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        target.addHandler(handler)
    return target
