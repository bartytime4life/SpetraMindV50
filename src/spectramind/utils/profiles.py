import os
from contextlib import contextmanager
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from .paths import ensure_dir


@contextmanager
def cpu_profile(out_path: str):
    """CPU profile via cProfile."""
    import cProfile
    import pstats

    ensure_dir(os.path.dirname(out_path))
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        with open(out_path, "w", encoding="utf-8") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.sort_stats("cumulative")
            ps.print_stats()


@contextmanager
def torch_profile(out_dir: str, use_cuda: bool = True, activities: Optional[list] = None):
    """Torch profiler context (if torch present); writes traces into out_dir."""
    if torch is None:
        yield
        return
    from torch.profiler import profile, record_function, ProfilerActivity  # type: ignore

    ensure_dir(out_dir)
    acts = activities or [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if (use_cuda and torch.cuda.is_available()) else [])
    with profile(activities=acts, on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir)) as prof:  # type: ignore
        yield
        prof.step()
