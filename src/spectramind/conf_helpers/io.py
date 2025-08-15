# SPDX-License-Identifier: MIT

"""I/O helpers for configuration files."""

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | os.PathLike, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=False)


def atomic_write(target_path: str | os.PathLike, data: str, encoding: str = "utf-8") -> None:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(target_path.parent), encoding=encoding) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(target_path)
