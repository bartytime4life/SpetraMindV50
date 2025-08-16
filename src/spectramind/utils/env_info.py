from typing import Dict, Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def detect_env() -> Dict[str, Any]:
    """Lightweight environment discovery (CUDA, torch)."""
    info: Dict[str, Any] = {}
    if torch is not None:
        info["torch_version"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info
