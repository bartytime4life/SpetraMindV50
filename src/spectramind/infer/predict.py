from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

import torch

from ..models.model_v50 import SpectraMindV50


def _to_device(obj: torch.Tensor, device: torch.device) -> torch.Tensor:
    return obj.to(device) if hasattr(obj, "to") else obj  # type: ignore[arg-type]


def predict_batch(ckpt_path: str, batch: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = SpectraMindV50(
        d_model=int(cfg.get("d_model", 128)), out_bins=int(cfg.get("out_bins", 283))
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        mu, sigma = model(
            _to_device(batch["fgs1"], device), _to_device(batch["airs"], device)
        )
        return {"mu": mu.cpu(), "sigma": sigma.cpu()}


def write_submission_csv(
    path: str, planet_ids: List[str], mu_tensor: torch.Tensor
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["planet_id"] + [f"bin_{i:03d}" for i in range(mu_tensor.shape[1])]
        )
        for pid, row in zip(planet_ids, mu_tensor.tolist()):
            writer.writerow([pid] + [f"{x:.6f}" for x in row])
