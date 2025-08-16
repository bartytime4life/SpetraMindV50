import hashlib
import json
import os
from typing import Any, Dict, Optional

from .jsonl import atomic_write_text


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any, algo: str = "blake2b", digest_size: int = 16) -> str:
    """Stable hash for arbitrary JSON-serializable obj."""
    canonical = _canonical_json(obj)
    if algo == "blake2b":
        h = hashlib.blake2b(canonical.encode("utf-8"), digest_size=digest_size)
    elif algo == "blake2s":
        h = hashlib.blake2s(canonical.encode("utf-8"), digest_size=min(32, digest_size))
    else:
        h = hashlib.sha256(canonical.encode("utf-8"))
    return h.hexdigest()


def hash_config(cfg: Any, out_path: Optional[str] = None) -> str:
    """Compute a stable config hash and optionally write it to a file."""
    h = stable_hash(cfg)
    if out_path:
        atomic_write_text(out_path, h + "\n")
    return h


def hash_file(path: str, algo: str = "blake2b", chunk_size: int = 1024 * 1024) -> str:
    """Hash file contents incrementally."""
    if algo == "blake2b":
        h = hashlib.blake2b(digest_size=16)
    elif algo == "blake2s":
        h = hashlib.blake2s(digest_size=16)
    else:
        h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(b: bytes, algo: str = "blake2b") -> str:
    if algo == "blake2b":
        return hashlib.blake2b(b, digest_size=16).hexdigest()
    elif algo == "blake2s":
        return hashlib.blake2s(b, digest_size=16).hexdigest()
    return hashlib.sha256(b).hexdigest()
