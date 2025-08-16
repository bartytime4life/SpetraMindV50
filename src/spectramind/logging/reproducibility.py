import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

def _git_info() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output([
            "git", "rev-parse", "HEAD"
        ], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    return {"commit": commit}

def capture_reproducibility_metadata(filename: str | Path) -> None:
    """Capture environment + git commit + config hash into ``filename``."""
    path = Path(filename)
    data = {
        "env": dict(os.environ),
        "git": _git_info(),
        "config_hash": os.environ.get("SPECTRAMIND_CONFIG_HASH", "unknown"),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
