import multiprocessing as mp
from typing import Callable, Iterable, List, Any, Sequence, Tuple


def cpu_count_safe(default: int = 2) -> int:
    try:
        n = mp.cpu_count()
        return max(1, n)
    except Exception:
        return max(1, default)


def chunked(seq: Sequence[Any], n: int) -> List[Sequence[Any]]:
    """Split a sequence into n nearly equal chunks (for simple parallelization)."""
    n = max(1, n)
    L = len(seq)
    size = (L + n - 1) // n
    return [seq[i : i + size] for i in range(0, L, size)]


def _worker(args: Tuple[Callable[[Any], Any], Any]) -> Any:
    """Unpack function and item for pool workers."""
    fn, item = args
    return fn(item)


def run_parallel(fn: Callable[[Any], Any], items: Sequence[Any], workers: int | None = None) -> List[Any]:
    """Simple process pool map with safe defaults (no daemon threads)."""
    if workers is None or workers <= 1:
        return [fn(x) for x in items]
    workers = min(cpu_count_safe(), workers)
    with mp.Pool(processes=workers) as pool:
        return pool.map(_worker, [(fn, x) for x in items])  # type: ignore[arg-type]
