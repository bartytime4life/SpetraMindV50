import os
import sys
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

from .common import logger

def confirm_guard(prompt: str = "Proceed? [y/N]: "):
    def deco(fn: Callable):
        @wraps(fn)
        def inner(*args, confirm: bool = False, **kwargs):
            if not confirm:
                resp = os.environ.get("SPECTRAMIND_CONFIRM", "").lower() or input(prompt).strip().lower()
                if resp not in ("y", "yes"):
                    logger.info("Aborted by user.")
                    sys.exit(1)
            return fn(*args, **kwargs)
        return inner
    return deco


def dry_run_guard(fn: Callable):
    @wraps(fn)
    def inner(*args, dry_run: bool = False, **kwargs):
        if dry_run:
            logger.info("[DRY-RUN] Command validated, not executing actions.")
            return 0
        return fn(*args, **kwargs)
    return inner


def require_file(path: Path, desc: Optional[str] = None):
    if not path.exists():
        logger.error("Missing %s: %s", desc or "file", path)
        sys.exit(2)
