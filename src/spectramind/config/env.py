"""
Environment helpers for SpectraMind V50 config system.

- get_device(): Return 'cuda' if available else 'cpu'
- get_env_var(): Read environment variables with default
"""

import os

def get_device() -> str:
    # Deferred import to avoid requiring torch at config import time
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        # If torch is not installed yet, fall back to env or cpu
        return os.environ.get("SPECTRAMIND_DEVICE", "cpu")

def get_env_var(key: str, default=None):
    return os.environ.get(key, default)
