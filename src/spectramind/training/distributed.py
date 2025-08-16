from __future__ import annotations

import logging
import os

try:
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    dist = None  # type: ignore


def init_distributed(backend: str = "nccl", timeout_seconds: int = 1800) -> bool:
    """
    Initialize torch.distributed if WORLD_SIZE > 1. Safe no-op otherwise.
    Returns True if distributed was initialized.
    """
    if torch is None or dist is None:
        return False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False

    if dist.is_available() and not dist.is_initialized():
        logging.info("Initializing torch.distributed backend=%s world_size=%d", backend, world_size)
        from datetime import timedelta

        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
        return True
    return dist.is_initialized()


def barrier() -> None:
    """Distributed barrier if initialized."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_rank() -> int:
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0


def is_main_process() -> bool:
    return get_rank() == 0
