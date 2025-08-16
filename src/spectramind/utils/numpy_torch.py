from contextlib import contextmanager
from typing import Any, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def to_numpy(x: Any):
    if np is None:
        raise ImportError("NumPy not installed")
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x if isinstance(x, np.ndarray) else np.asarray(x)


def to_tensor(x: Any, device: Optional[str] = None, dtype: Optional[str] = None):
    if torch is None:
        raise ImportError("PyTorch not installed")
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    if dtype:
        t = t.to(getattr(torch, dtype))
    if device:
        t = t.to(device)
    return t


def move_to_device(obj: Any, device: str):
    if torch is None:
        raise ImportError("PyTorch not installed")
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    return obj


def amp_dtype_from_str(s: Optional[str]):
    if torch is None:
        return None
    if not s:
        return None
    s = s.lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "single"):
        return torch.float32
    return None


@contextmanager
def autocast_if_available(enabled: bool = True, dtype: Optional[str] = None):
    if torch is None or not enabled:
        yield
        return
    amp_dtype = amp_dtype_from_str(dtype)
    if amp_dtype is None:
        from contextlib import nullcontext
        with nullcontext():
            yield
        return
    from torch.cuda.amp import autocast  # type: ignore
    with autocast(dtype=amp_dtype):
        yield
