from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch may not be present
    torch = None  # type: ignore


@dataclass
class RunMeta:
    """Immutable run metadata, captured once at startup and reused everywhere."""

    cli_version: str
    config_hash: str
    start_ts: float
    pid: int
    host: str
    cwd: str
    python: str
    torch: Optional[str]
    seed: int


def make_dirs(*paths: str | os.PathLike) -> None:
    """Create directories if they don't already exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set deterministic seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    """Compute a short stable hash of a nested dict-like Hydra config."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def capture_run_meta(cfg: Dict[str, Any], cli_version: str, seed: int) -> RunMeta:
    """Capture one-shot run metadata."""
    return RunMeta(
        cli_version=cli_version,
        config_hash=compute_config_hash(cfg),
        start_ts=time.time(),
        pid=os.getpid(),
        host=socket.gethostname(),
        cwd=os.getcwd(),
        python=sys.version.replace("\n", " "),
        torch=(None if torch is None else torch.__version__),
        seed=seed,
    )


def write_debug_log(meta: RunMeta, log_path: str = "v50_debug_log.md") -> None:
    """Append a single line to v50_debug_log.md with run metadata."""
    line = (
        f"| {datetime.fromtimestamp(meta.start_ts).isoformat()} "
        f"| cfg={meta.config_hash} | ver={meta.cli_version} | pid={meta.pid} "
        f"| host={meta.host} | py={meta.python.split()[0]} | torch={meta.torch} | seed={meta.seed} |\n"
    )
    header = (
        "# SpectraMind V50 – CLI Call Log (append-only)\n"
        "| timestamp | cfg_hash | version | pid | host | python | torch | seed |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    p = Path(log_path)
    if not p.exists():
        p.write_text(header)
    with p.open("a", encoding="utf-8") as f:
        f.write(line)


def human_ts(ts: Optional[float] = None) -> str:
    """Human-friendly timestamp string for filenames and logs."""
    return datetime.fromtimestamp(ts or time.time()).strftime("%Y%m%d-%H%M%S")


def get_device(prefer: str = "cuda") -> Tuple[str, int]:
    """Choose device and visible device index.

    Returns
    -------
    (device_str, device_index)
    """
    if torch is None:
        return "cpu", -1
    if prefer == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return f"cuda:{idx}", idx
    return "cpu", -1


def save_checkpoint(state: Dict[str, Any], ckpt_dir: str, tag: str, keep_last: int = 5) -> str:
    """Save a checkpoint and keep only the most recent N."""
    make_dirs(ckpt_dir)
    fname = f"ckpt_{tag}_{human_ts()}.pt"
    fpath = str(Path(ckpt_dir) / fname)
    if torch is None:
        # Fallback to numpy-friendly save
        Path(fpath).write_bytes(json.dumps(state).encode("utf-8"))
    else:
        torch.save(state, fpath)
    # Cleanup old
    ckpts = sorted(
        Path(ckpt_dir).glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for old in ckpts[keep_last:]:
        try:
            old.unlink()
        except Exception:
            pass
    return fpath


def load_checkpoint(fpath: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint robustly whether torch is present or not."""
    if torch is None:
        return json.loads(Path(fpath).read_text(encoding="utf-8"))
    return torch.load(fpath, map_location=map_location or "cpu")


def export_manifest(
    artifacts: Dict[str, str | int | float | Dict[str, Any]], out_path: str
) -> None:
    """Write a JSON manifest with artifact locations, hashes, and metadata."""
    Path(out_path).write_text(
        json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8"
    )

