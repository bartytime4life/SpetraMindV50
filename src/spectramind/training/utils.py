from __future__ import annotations

import dataclasses
import functools
import getpass
import hashlib
import importlib
import json
import logging
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover
    OmegaConf = None  # type: ignore


def rank_zero_only(fn: Callable) -> Callable:
    """Decorator to run the function only on the main process (rank 0)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if get_global_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapper


def get_global_rank() -> int:
    """Return distributed rank if available, else 0."""
    if "RANK" in os.environ:
        try:
            return int(os.environ["RANK"])
        except Exception:
            return 0
    if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def world_size() -> int:
    """Return world size if available, else 1."""
    if "WORLD_SIZE" in os.environ:
        try:
            return int(os.environ["WORLD_SIZE"])
        except Exception:
            return 1
    if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def import_from_string(path: str) -> Callable:
    """
    Import an object from a string path "module.submodule:object".
    Raises informative error if not found.
    """
    if ":" not in path:
        raise ValueError(f"Expected 'module.submodule:object' format, got '{path}'")
    module_path, obj_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, obj_name):
        raise ImportError(f"Object '{obj_name}' not found in module '{module_path}'")
    return getattr(module, obj_name)


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def stable_serialize_config(cfg: Any) -> str:
    """
    Produce a normalized JSON string of the config for hashing and logs.
    Supports OmegaConf configs and standard dicts.
    """
    if OmegaConf is not None and "DictConfig" in str(type(cfg)):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    else:
        cfg_dict = cfg
    return json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"))


def config_hash(cfg: Any) -> str:
    """Return a short, stable sha256 hash of the given config."""
    s = stable_serialize_config(cfg).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:12]


def get_git_hash(default: str = "unknown") -> str:
    """Return the current git commit hash if available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return default


def env_info() -> Dict[str, Any]:
    """Capture environment information for reproducibility logs."""
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "time_utc": datetime.utcnow().isoformat() + "Z",
        "process_pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "world_size": world_size(),
        "rank": get_global_rank(),
    }
    if torch is not None:
        info.update(
            {
                "torch_version": getattr(torch, "__version__", "unknown"),
                "cuda_available": torch.cuda.is_available(),
                "cudnn_enabled": getattr(torch.backends, "cudnn", None) is not None,
                "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        )
        if torch.cuda.is_available():
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def ensure_dir(path: Path | str) -> Path:
    """Ensure directory exists and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_run_dir(base: Path | str, cfg_hash: str) -> Path:
    """
    Create a timestamped run directory under base using the config hash.
    Example: runs/2025-08-16_12-03-10_hashABC123
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ensure_dir(Path(base) / f"{ts}_{cfg_hash}")
    return run_dir


def to_device(batch: Any, device: Any) -> Any:
    """Recursively move tensors in batch to device."""
    if torch is None:
        return batch
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    if isinstance(batch, (torch.Tensor,)):
        return batch.to(device, non_blocking=True)
    return batch


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    """Flatten nested dictionaries using `sep` as a delimiter."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def write_json(path: Path | str, data: Dict[str, Any]) -> None:
    """Write dict to JSON with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: Path | str) -> Dict[str, Any]:
    """Read JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
