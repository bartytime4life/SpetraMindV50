from typing import Any, Iterable, Sequence, Tuple


def require_keys(d: dict, keys: Iterable[str]) -> dict:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")
    return d


def ensure_type(x: Any, typ: type, name: str = "value") -> Any:
    if not isinstance(x, typ):
        raise TypeError(f"{name} must be {typ}, got {type(x)}")
    return x


def ensure_shape(arr: Any, shape: Sequence[int], name: str = "array") -> Any:
    """Minimal shape validator for NumPy arrays or Torch tensors."""
    try:
        actual = tuple(int(s) for s in arr.shape)
    except Exception:
        raise TypeError(f"{name} must have .shape")
    if len(actual) != len(shape):
        raise ValueError(f"{name} rank mismatch: expected {shape}, got {actual}")
    for i, (a, b) in enumerate(zip(actual, shape)):
        if b != -1 and a != b:
            raise ValueError(f"{name} shape mismatch at dim {i}: expected {shape}, got {actual}")
    return arr


def validate_numeric_range(x: Any, lo: float | None = None, hi: float | None = None, name: str = "value") -> Any:
    v = float(x)
    if lo is not None and v < lo:
        raise ValueError(f"{name}={v} < lo={lo}")
    if hi is not None and v > hi:
        raise ValueError(f"{name}={v} > hi={hi}")
    return x
