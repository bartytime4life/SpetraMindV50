"""Symbolic loss computation based on rule evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - torch is optional
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = Any  # type: ignore

from .symbolic_logic_engine import SymbolicEvalResult, SymbolicLogicEngine
from .symbolic_utils import append_debug_log, now_utc_iso

DEBUG_LOG = "logs/v50_debug_log.md"


@dataclass
class SymbolicLossOutput:
    total_loss: float
    per_rule: Dict[str, float]
    masks: Dict[str, Optional[np.ndarray]]


class SymbolicLoss:
    """Compute symbolic losses on μ (and optional σ)."""

    def __init__(self, engine: SymbolicLogicEngine, reduction: str = "sum") -> None:
        self.engine = engine
        self.reduction = reduction

    def __call__(
        self, mu, sigma=None, metadata: Optional[Dict[str, Any]] = None
    ) -> SymbolicLossOutput:
        is_torch = (torch is not None) and isinstance(mu, Tensor)
        if is_torch:
            mu_np = mu.detach().cpu().numpy()
            sg_np = None if sigma is None else sigma.detach().cpu().numpy()
        else:
            mu_np = np.asarray(mu)
            sg_np = None if sigma is None else np.asarray(sigma)

        results: List[SymbolicEvalResult] = self.engine.evaluate(
            mu_np, sg_np, metadata=metadata
        )
        per_rule = {r.rule_name: float(r.magnitude) for r in results}
        masks = {r.rule_name: r.mask for r in results}
        total = float(sum(per_rule.values()))
        append_debug_log(
            DEBUG_LOG,
            f"[{now_utc_iso()}] SymbolicLoss: total={total:.6f} rules={len(per_rule)}",
        )

        if is_torch:
            total_t = torch.tensor(
                total, dtype=mu.dtype, device=mu.device, requires_grad=False
            )
            return SymbolicLossOutput(
                total_loss=float(total_t.item()), per_rule=per_rule, masks=masks
            )
        return SymbolicLossOutput(total_loss=total, per_rule=per_rule, masks=masks)
