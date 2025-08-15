"""
Copyright (c) 2025 SpectraMind.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


def _run(cmd: List[str], cwd: Optional[str | Path] = None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _git_info(root: Optional[str | Path] = None) -> Dict[str, Any]:
    info = {
        "git_present": False,
        "branch": "",
        "commit": "",
        "dirty": False,
        "remote": "",
        "describe": "",
    }
    if _run(["git", "--version"]):
        info["git_present"] = True
        info["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
        info["commit"] = _run(["git", "rev-parse", "HEAD"], cwd=root)
        info["describe"] = _run(["git", "describe", "--always", "--dirty", "--tags"], cwd=root)
        status = _run(["git", "status", "--porcelain"], cwd=root)
        info["dirty"] = bool(status)
        info["remote"] = _run(["git", "remote", "-v"], cwd=root)
    return info


def _pip_freeze() -> List[str]:
    try:
        out = _run([sys.executable, "-m", "pip", "freeze"])
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def _cuda_version() -> str:
    nv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if nv:
        return f"driver={nv}"
    return ""


def capture_env_git_snapshot(repo_root: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Return a JSON-serializable dictionary capturing OS, Python, CUDA (if present), env vars subset,
    and Git metadata. Intended for embedding into run manifests or JSONL events.
    """
    env_subset = {k: os.environ.get(k, "") for k in [
        "USER", "HOSTNAME", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "VIRTUAL_ENV", "CONDA_PREFIX",
        "MLFLOW_TRACKING_URI", "WANDB_PROJECT", "WANDB_ENTITY",
    ]}
    snapshot = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": sys.version.replace("\n", " "),
        },
        "cuda": _cuda_version(),
        "env": env_subset,
        "git": _git_info(repo_root),
        "pip_freeze": _pip_freeze()[:400],  # cap to keep JSON small; full freeze can be large
    }
    return snapshot
