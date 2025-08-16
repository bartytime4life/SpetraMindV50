from typing import Any, Dict

try:
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    dist = None  # type: ignore


def ddp_available() -> bool:
    return (torch is not None) and (dist is not None)


def init_distributed(backend: str = "nccl", init_method: str | None = None) -> None:
    if not ddp_available():
        return
    if dist.is_initialized():
        return
    if init_method is None:
        init_method = "env://"
    dist.init_process_group(backend=backend, init_method=init_method)


def ddp_rank() -> int:
    if not ddp_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def ddp_world_size() -> int:
    if not ddp_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def ddp_is_master() -> bool:
    return ddp_rank() == 0


def ddp_barrier() -> None:
    if ddp_available() and dist.is_initialized():
        dist.barrier()


def ddp_reduce_dict(metrics: Dict[str, Any], average: bool = True) -> Dict[str, Any]:
    """All-reduce a dict of scalar tensors across ranks."""
    if not ddp_available() or not dist.is_initialized() or ddp_world_size() == 1:
        return metrics
    out = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            t = v.clone().float()
        else:
            t = torch.tensor(float(v), dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if average:
            t = t / ddp_world_size()
        out[k] = t.item()
    return out
