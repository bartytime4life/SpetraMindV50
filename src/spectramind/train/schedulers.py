from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch.optim import Optimizer
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Optimizer = object  # type: ignore


def cosine_with_warmup(
    optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0
):
    """Build a cosine annealing with linear warmup scheduler.

    If torch is not available, this returns a dummy object with ``step()`` method.
    """
    if torch is None:
        class _Dummy:
            def step(self):  # noqa: D401
                """No-op scheduler used when torch is unavailable."""
                return

        return _Dummy()

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
        return max(min_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

