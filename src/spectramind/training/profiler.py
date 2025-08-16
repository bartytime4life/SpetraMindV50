from __future__ import annotations

from contextlib import contextmanager

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


@contextmanager
def null_profiler():
    yield


def get_profiler(enabled: bool = False):
    if not enabled or not _HAS_TORCH:
        return null_profiler()
    # Lightweight CPU profiler placeholder; extend as needed for torch.profiler
    return null_profiler()
