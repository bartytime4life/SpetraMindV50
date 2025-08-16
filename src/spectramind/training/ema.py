from __future__ import annotations

from typing import Dict

import torch


class ExponentialMovingAverage:
    """
    Simple EMA of model parameters. Use .apply_to(model) to copy EMA weights to model.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.collected: Dict[str, torch.Tensor] = {}
        self.register(model)

    def register(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * p.detach() + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_to(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)
