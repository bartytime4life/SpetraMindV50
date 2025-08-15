import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict


def _safe(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "N/A"


def capture_git_env_state(
    output_path: Path = Path("logs/git_env_snapshot.json"),
) -> Dict[str, Any]:
    """
    Captures git commit/branch/dirty state plus Python/OS and selected environment variables.
    Returns the dict and writes JSON to logs/git_env_snapshot.json.
    """
    commit = _safe(["git", "rev-parse", "HEAD"])
    branch = _safe(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _safe(["git", "status", "--porcelain"])
    dirty = bool(status and status != "N/A")

    env_vars = {}
    for k, v in os.environ.items():
        if k.startswith(
            (
                "CUDA",
                "PYTHON",
                "PATH",
                "VIRTUAL_ENV",
                "CONDA",
                "HYDRA",
                "WANDB",
                "MLFLOW",
            )
        ):
            env_vars[k] = v

    payload = {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "env_vars": env_vars,
    }
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload
