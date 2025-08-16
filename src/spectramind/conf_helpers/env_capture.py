from __future__ import annotations

import datetime
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict


def _try(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return "unavailable"


def _git_info() -> Dict[str, str]:
    return {
        "commit": _try(["git", "rev-parse", "HEAD"]),
        "branch": _try(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status": _try(["git", "status", "--porcelain"]),
        "remote": _try(["git", "remote", "-v"]),
        "tag": _try(["git", "describe", "--tags", "--always"]),
    }


def _gpu_info() -> Dict[str, str]:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return {"nvidia_smi": "not found"}
    return {
        "nvidia_smi_top": _try([nvsmi, "-L"]),
        "nvidia_smi_detail": _try([nvsmi]),
    }


def _pip_freeze() -> str:
    return _try([shutil.which("python") or "python", "-m", "pip", "freeze"])


def capture_environment() -> Dict[str, Any]:
    """
    Lightweight capture (safe for unit tests).
    """
    return {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "hydra_full_error": os.environ.get("HYDRA_FULL_ERROR", ""),
        "git": _git_info(),
    }


def capture_environment_detailed() -> Dict[str, Any]:
    """
    Detailed capture (includes GPU and pip freeze). Use for dashboard audits.
    """
    env = capture_environment()
    env["gpu"] = _gpu_info()
    env["pip_freeze"] = _pip_freeze()
    return env


def log_environment(out_path: str, detailed: bool = False) -> Dict[str, Any]:
    env = capture_environment_detailed() if detailed else capture_environment()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(env, f, indent=2)
    return env
