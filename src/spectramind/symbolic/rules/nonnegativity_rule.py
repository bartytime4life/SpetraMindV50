# SPDX-License-Identifier: Apache-2.0

"""Enforce non-negativity of spectra."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from .base_rule import SymbolicRule


class NonNegativityRule(SymbolicRule):
    """Penalise negative values in ``mu`` via ``relu(-mu)``."""

    def __init__(self, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name="nonnegativity", weight=weight, **kwargs)

    def evaluate_map(
        self,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> Tensor:
        return torch.relu(-mu)


__all__ = ["NonNegativityRule"]
