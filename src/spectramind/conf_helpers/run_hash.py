import hashlib
import json
from typing import Any, Dict


def config_hash(cfg: Dict[str, Any]) -> str:
    """Return a short deterministic hash for a configuration dictionary."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
