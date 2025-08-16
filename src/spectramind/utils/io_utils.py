import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from .jsonl import atomic_write_text


def path_exists(path: str) -> bool:
    return os.path.exists(path)


def safe_mkdirs(path: str) -> str:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str, pretty: bool = True) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False) if pretty else json.dumps(obj, ensure_ascii=False)
    atomic_write_text(path, text)


def load_yaml(path: str) -> Any:
    if yaml is None:
        raise ImportError("PyYAML not installed")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Any, path: str) -> None:
    if yaml is None:
        raise ImportError("PyYAML not installed")
    text = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    atomic_write_text(path, text)


def load_csv(path: str, has_header: bool = True) -> List[List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if has_header and rows:
            return rows  # include header row for generality
        return rows


def save_csv(rows: Iterable[Iterable[Any]], path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(list(r))


def save_npz(path: str, **arrays: Any) -> None:
    if np is None:
        raise ImportError("NumPy not installed")
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> Dict[str, Any]:
    if np is None:
        raise ImportError("NumPy not installed")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_torch(obj: Any, path: str) -> None:
    if torch is None:
        raise ImportError("PyTorch not installed")
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    torch.save(obj, path)


def load_torch(path: str, map_location: Optional[str] = None) -> Any:
    if torch is None:
        raise ImportError("PyTorch not installed")
    return torch.load(path, map_location=map_location or "cpu")
