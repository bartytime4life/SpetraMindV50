import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Timer:
    """High-precision timer (monotonic) usable as context manager."""
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.elapsed = time.perf_counter() - self.start


class Stopwatch:
    """Manual start/stop stopwatch that can be resumed."""

    def __init__(self) -> None:
        self._start = None
        self.elapsed = 0.0

    def start(self) -> None:
        if self._start is None:
            self._start = time.perf_counter()

    def stop(self) -> None:
        if self._start is not None:
            self.elapsed += time.perf_counter() - self._start
            self._start = None

    def reset(self) -> None:
        self._start = None
        self.elapsed = 0.0


@contextmanager
def time_block(name: str) -> Iterator[float]:
    """Context manager to time a block and yield elapsed seconds at exit."""
    t0 = time.perf_counter()
    try:
        yield t0
    finally:
        dt = time.perf_counter() - t0
        print(f"[time_block] {name}: {dt:.6f}s")
