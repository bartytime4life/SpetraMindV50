import datetime
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pragma: no cover - best effort
        return None


def capture_git_env() -> Dict[str, Any]:
    """Return a dictionary with git and environment metadata."""
    info: Dict[str, Any] = {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status": _run(["git", "status", "--porcelain"]),
        "git_remote": _run(["git", "remote", "-v"]),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "ts_iso": datetime.datetime.utcnow().isoformat() + "Z",
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                ("SM", "KAGGLE", "WANDB", "MLFLOW", "HYDRA_", "CUDA", "PYTORCH")
            )
        },
    }
    return info


def dump_git_env_json(path: str) -> None:
    """Write :func:`capture_git_env` to ``path`` as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(capture_git_env(), fh, indent=2, sort_keys=True)
