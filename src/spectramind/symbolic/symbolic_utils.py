"""Utility helpers for the symbolic subsystem."""

from __future__ import annotations

import hashlib
import importlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML might not be available
    yaml = None


def set_global_seed(seed: int = 42) -> None:
    """Set RNG seeds across libraries for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy may be absent
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - torch may be absent
        pass


def now_utc_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_config_like(obj: Any) -> str:
    """Generate a short hash for dict/list/str configuration objects."""
    if isinstance(obj, (dict, list)):
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    else:
        payload = str(obj)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_yaml(path: str | Path, data: Any) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write YAML files")
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_yaml(path: str | Path) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML files")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_plotly_html(fig: Any, path: str | Path) -> None:
    """Save a Plotly figure to HTML if Plotly is installed."""
    try:
        ensure_dir(Path(path).parent)
        fig.write_html(str(path), include_plotlyjs="cdn")
    except Exception:  # pragma: no cover - plotly may be absent
        pass


def append_debug_log(path: str | Path, text: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + ("\n" if not text.endswith("\n") else ""))


def lazy_import(name: str):
    """Import a module lazily, returning the module or raising ImportError."""
    return importlib.import_module(name)
