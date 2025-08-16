import json
import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - OmegaConf optional at runtime
    OmegaConf = None  # type: ignore

def seed_everything(seed: int = 1337, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for determinism.
    Optionally enable cuDNN deterministic behavior (may impact performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]

def get_device(prefer: str = "cuda") -> torch.device:
    """
    Select a torch.device with preference ordering.
    prefer: "cuda" | "mps" | "cpu"
    """
    prefer = prefer.lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in ("mps", "metal"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            return torch.device("mps")
    return torch.device("cpu")

def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters. If trainable_only=True, only parameters with requires_grad.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def git_commit_hash() -> Optional[str]:
    """
    Return the current Git commit hash if within a Git repo; else None.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def dump_yaml_safely(obj: Any, path: str) -> None:
    """
    Dump a Python object to path as YAML if OmegaConf is available, else JSON.
    Ensures we can always write a serializable snapshot.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if OmegaConf is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.create(obj)))  # type: ignore
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

@dataclass
class RunContext:
    """
    Minimal run context describing runtime environment for logging & reproducibility.
    """
    hostname: str
    pid: int
    python: str
    argv: str
    cwd: str
    time_utc: float
    git_hash: Optional[str]

    @staticmethod
    def capture() -> "RunContext":
        return RunContext(
            hostname=socket.gethostname(),
            pid=os.getpid(),
            python=sys.version.replace("\n", " "),
            argv=" ".join(sys.argv),
            cwd=str(Path.cwd()),
            time_utc=time.time(),
            git_hash=git_commit_hash(),
        )

def ensure_dir(path: str) -> None:
    """
    Create directory if missing; no error if exists.
    """
    os.makedirs(path, exist_ok=True)
