"""Git & environment capture for reproducibility metadata."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class GitInfo:
    branch: str = ""
    commit: str = ""
    is_dirty: bool = False
    remote_url: str = ""


@dataclass
class EnvInfo:
    python: str = sys.version.replace("\n", " ")
    platform: str = platform.platform()
    cuda: str = ""
    torch: str = ""
    numpy: str = ""


def _run_cmd(cmd: str) -> str:
    try:
        out = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=3
        )
        return out.strip()
    except Exception:
        return ""


def get_git_info() -> GitInfo:
    branch = _run_cmd("git rev-parse --abbrev-ref HEAD")
    commit = _run_cmd("git rev-parse HEAD")
    status = _run_cmd("git status --porcelain")
    remote = _run_cmd("git remote get-url origin")
    return GitInfo(branch=branch, commit=commit, is_dirty=(status != ""), remote_url=remote)


def get_env_info() -> EnvInfo:
    cuda = os.getenv("CUDA_VERSION", "")
    torch_v = ""
    try:
        import torch  # type: ignore

        torch_v = torch.__version__
        if not cuda:
            cuda = getattr(torch.version, "cuda", "")
    except Exception:
        pass
    numpy_v = ""
    try:
        import numpy as np  # type: ignore

        numpy_v = np.__version__
    except Exception:
        pass
    return EnvInfo(cuda=cuda, torch=torch_v, numpy=numpy_v)


def reproducibility_snapshot(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "git": asdict(get_git_info()),
        "env": asdict(get_env_info()),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }
    if extra:
        snap.update(extra)
    return snap
