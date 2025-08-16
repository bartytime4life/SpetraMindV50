import os
import pickle
from functools import wraps
from typing import Callable, Any, Dict, Optional

from .paths import get_default_paths, ensure_dir
from .hashing import stable_hash


def disk_cache(name: str, version: str = "v1"):
    """Disk-caching decorator keyed by function name + args hash."""
    base = os.path.join(get_default_paths()["artifacts_dir"], "cache", name, version)
    ensure_dir(base)

    def deco(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = stable_hash({"args": repr(args), "kwargs": repr(kwargs)})
            path = os.path.join(base, f"{key}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
            out = fn(*args, **kwargs)
            with open(path, "wb") as f:
                pickle.dump(out, f)
            return out

        return wrapper

    return deco
