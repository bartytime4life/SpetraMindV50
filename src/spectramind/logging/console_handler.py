import logging
import sys
from typing import Optional

def setup_console_logging(*, level: int = logging.INFO, logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Configure a simple console handler on the given logger (root by default)."""
    target = logger or logging.getLogger()
    target.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in target.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        target.addHandler(handler)
    return target
