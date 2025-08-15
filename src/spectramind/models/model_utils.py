# Utility helpers for SpectraMind V50 models: initialization, summaries, checkpoints, seeding.

from __future__ import annotations
import os
import io
import json
import math
import random
import hashlib
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable full determinism where feasible
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False, warn_only=True)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    params = [p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)]
    return int(sum(params))


def summarize_module(module: nn.Module) -> str:
    lines = []
    total = 0
    for name, p in module.named_parameters():
        n = p.numel()
        req = "*" if p.requires_grad else "-"
        lines.append(f"{name:60s} {n:12d} {req}")
        total += n
    lines.append("-" * 80)
    lines.append(f"{'TOTAL':60s} {total:12d}")
    return "\n".join(lines)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(model: nn.Module, path: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    payload["sha256"] = _sha256_file(path)
    with open(path + ".json", "w") as f:
        json.dump({"sha256": payload["sha256"], "meta": payload["extra"]}, f, indent=2)
    return payload


def load_checkpoint(model: nn.Module, path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload
