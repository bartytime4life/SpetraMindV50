from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class GenericGaussianProcessor:
    """
    Generic step processor assuming the model returns a dict with 'mu' and 'sigma'
    (or a tuple (mu, sigma)), and batch provides inputs 'x' and target 'y'.

    This class abstracts the per-batch training/validation step logic used by TrainerBase.
    """
    loss_module: torch.nn.Module  # Expected to return a dict with {'total': ..., ...}

    def __call__(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        out = model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            mu, sigma = out[0], out[1]
        elif isinstance(out, dict) and "mu" in out and "sigma" in out:
            mu, sigma = out["mu"], out["sigma"]
        else:
            raise ValueError("Model output must be (mu, sigma) tuple or dict with keys 'mu' and 'sigma'.")

        losses = self.loss_module(mu, sigma, y)
        total = losses["total"]
        scalars = {k: float(v.detach().item()) for k, v in losses.items()}
        return total, scalars
