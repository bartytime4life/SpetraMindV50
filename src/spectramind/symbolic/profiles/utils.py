"""General utilities: repo root discovery, YAML I/O, deep merge, SHA256, YAML dir loader."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover - defer error until first use
    yaml = None


def find_repo_root(start: Path | None = None) -> Path:
    """Heuristic to find repository root: ascend until a .git or pyproject.toml or .repo-root
    marker. Fallback to current working directory."""
    if start is None:
        start = Path.cwd()
    p = start.resolve()
    while True:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / ".repo-root").exists():
            return p
        if p.parent == p:
            return start.resolve()
        p = p.parent


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for this operation. Please install with: pip install pyyaml"
        )


def safe_load_yaml(text_or_path: str | Path):
    """Load YAML from a string path or text content (auto-detect by path existence)."""
    _require_yaml()
    if isinstance(text_or_path, (str, Path)) and Path(str(text_or_path)).exists():
        with Path(str(text_or_path)).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return yaml.safe_load(str(text_or_path))


def safe_dump_yaml(obj: Any) -> str:
    _require_yaml()
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dict b into a (non-destructive), with lists replaced by b."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def sha256_of_text(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def load_all_yaml_files_in_dir(dir_path: Path) -> List[Tuple[Path, Any]]:
    """Load all .yaml/.yml under dir_path (non-recursive). Returns list of (path, data).
    Missing directory returns empty list."""
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    items: List[Tuple[Path, Any]] = []
    for ext in (".yaml", ".yml"):
        for p in sorted(dir_path.glob(f"*{ext}")):
            data = safe_load_yaml(p)
            items.append((p, data))
    return items
