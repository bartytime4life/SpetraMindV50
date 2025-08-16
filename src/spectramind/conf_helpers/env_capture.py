"""Environment capture utilities."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict


def capture_environment() -> Dict[str, Any]:
    """Capture basic environment information."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "env": dict(os.environ),
    }


def log_environment(path: str | Path) -> Dict[str, Any]:
    """Capture environment information and persist it to ``path`` as JSON."""
    data = capture_environment()
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data
