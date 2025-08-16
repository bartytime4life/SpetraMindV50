import io
import json
import os
import tempfile
from typing import Any, Dict, Iterable, Iterator, Optional


def atomic_write_bytes(path: str, data: bytes, mode: str = "wb") -> None:
    """Atomically write bytes to a file using a temporary file and os.replace."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".tmp.", suffix=".atomic")
    try:
        with os.fdopen(fd, mode) as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """Atomically write text to a file."""
    atomic_write_bytes(path, text.encode(encoding), "wb")


def jsonl_append(path: str, obj: Dict[str, Any], ensure_ascii: bool = False) -> None:
    """Append a single JSON object as a line to a .jsonl file, creating parent directories if needed."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with io.open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=ensure_ascii) + "\n")


def jsonl_iter(path: str) -> Iterator[Dict[str, Any]]:
    """Iterate over JSONL lines, yielding dicts. Skips malformed lines."""
    if not os.path.exists(path):
        return
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
