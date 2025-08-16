import os
import json
import platform
import random
import subprocess
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from .paths import get_default_paths, ensure_dir
from .jsonl import atomic_write_text
from .hashing import stable_hash


def set_all_seeds(seed: int = 1337, deterministic: bool = True) -> None:
    """Set seeds across python, numpy, torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def collect_git_info(root: Optional[str] = None) -> Dict[str, Any]:
    """Collect minimal Git info (commit, branch, dirty)."""
    cwd = root or get_default_paths()["root"]
    commit = _run(["git", "-C", cwd, "rev-parse", "HEAD"])
    branch = _run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "-C", cwd, "status", "--porcelain"])
    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "dirty": bool(status),
    }


def summarize_packages() -> Dict[str, Any]:
    """Return a lightweight summary of key package versions (avoids pip freeze cost)."""
    vers = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if torch is not None:
        vers["torch"] = str(torch.__version__)
        vers["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            vers["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
    if np is not None:
        vers["numpy"] = str(np.__version__)
    try:
        import omegaconf  # type: ignore

        vers["omegaconf"] = str(omegaconf.__version__)
    except Exception:
        pass
    try:
        import hydra  # type: ignore

        vers["hydra"] = getattr(hydra, "__version__", "unknown")
    except Exception:
        pass
    return vers


def collect_env_info(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect environment info suitable for run snapshots and audit trails."""
    env = {
        "env_vars_subset": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("PYTORCH_")
            or k.startswith("CUDA_")
            or k in (
                "HF_HOME",
                "WANDB_MODE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
        "versions": summarize_packages(),
    }
    if extra:
        env["extra"] = extra
    return env


def snapshot_run_info(
    cfg_like: Any,
    seed: int,
    out_dir: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a deterministic snapshot of the run (config hash, git, env) and write to artifacts."""
    paths = get_default_paths()
    run_dir = ensure_dir(out_dir or paths["run_info_dir"])
    run = {
        "config_hash": stable_hash(cfg_like),
        "git": collect_git_info(paths["root"]),
        "env": collect_env_info(),
        "seed": seed,
        "notes": notes or "",
    }
    out_path = os.path.join(run_dir, "run_info.json")
    atomic_write_text(out_path, json.dumps(run, indent=2, ensure_ascii=False))
    return run
