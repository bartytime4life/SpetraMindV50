from __future__ import annotations

import math
import os
import time
from typing import Any, Dict

import torch
import torch.utils.data as data

from ..data.datasets import AirsDataset, Fgs1Dataset, train_val_split
from ..models.model_v50 import SpectraMindV50
from ..utils.git_env import dump_git_env_json
from ..utils.hydra_safe import write_run_manifest
from ..utils.logging_setup import build_logger, log_event
from ..utils.mlflow_wandb import (
    try_mlflow_log_metrics,
    try_mlflow_log_params,
    try_mlflow_start,
    try_wandb_init,
)
from .losses import gll_mean


def seed_all(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JointDataset(data.Dataset):
    def __init__(self, base: AirsDataset, aux: Fgs1Dataset, idxs: list[int]):
        self.base, self.aux, self.idxs = base, aux, idxs

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.idxs)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        j = self.idxs[i]
        r1 = self.base[j]
        r2 = self.aux[j]
        return {
            "airs": r1["airs"],
            "fgs1": r2["fgs1"],
            "mu": r1["mu"],
            "sigma": r1["sigma"],
            "meta": r1["meta"],
        }


def collate(batch: list[Dict[str, Any]]):
    import numpy as np

    airs = torch.tensor(np.stack([b["airs"] for b in batch], 0))
    fgs1 = torch.tensor(np.stack([b["fgs1"] for b in batch], 0))
    mu = torch.tensor(np.stack([b["mu"] for b in batch], 0))
    sigma = torch.tensor(np.stack([b["sigma"] for b in batch], 0))
    return airs, fgs1, mu, sigma


def train_v50(config: Dict[str, Any]) -> Dict[str, Any]:
    logger, jsonl, paths = build_logger(level=20)
    log_event(jsonl, "start.train", {"config": config})
    run_dir = os.path.join("runs", f"train_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    dump_git_env_json(os.path.join(run_dir, "env_git.json"))
    write_run_manifest(
        os.path.join(run_dir, "manifest.json"), {"config": config, "paths": paths}
    )

    seed_all(int(config.get("seed", 1337)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_airs = AirsDataset(
        root=config.get("data_root", "."), split="train", seed=config.get("seed", 1337)
    )
    ds_fgs1 = Fgs1Dataset(
        root=config.get("data_root", "."), split="train", seed=config.get("seed", 1337)
    )
    n = min(len(ds_airs), len(ds_fgs1))
    train_idx, val_idx = train_val_split(
        n, val_frac=float(config.get("val_frac", 0.1)), seed=config.get("seed", 1337)
    )

    train_ds = JointDataset(ds_airs, ds_fgs1, train_idx)
    val_ds = JointDataset(ds_airs, ds_fgs1, val_idx)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    model = SpectraMindV50(
        d_model=int(config.get("d_model", 128)),
        out_bins=int(config.get("out_bins", 283)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 3e-4)))
    epochs = int(config.get("epochs", 2))

    mlrun = try_mlflow_start("train_v50", tags={"phase": "train"})
    if mlrun:
        try_mlflow_log_params(config)
    wandb_run = try_wandb_init("train_v50", config=config)

    best_val = math.inf
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        nb = 0
        for airs, fgs1, mu, sigma in train_loader:
            airs, fgs1, mu, sigma = (
                airs.to(device),
                fgs1.to(device),
                mu.to(device),
                sigma.to(device),
            )
            opt.zero_grad()
            pred_mu, pred_sigma = model(fgs1, airs)
            loss = gll_mean(pred_mu, pred_sigma, mu)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            nb += 1
        tr_loss /= max(nb, 1)

        model.eval()
        vl = 0.0
        nbv = 0
        with torch.no_grad():
            for airs, fgs1, mu, sigma in val_loader:
                airs, fgs1, mu, sigma = (
                    airs.to(device),
                    fgs1.to(device),
                    mu.to(device),
                    sigma.to(device),
                )
                pm, ps = model(fgs1, airs)
                val_loss = gll_mean(pm, ps, mu)
                vl += val_loss.item()
                nbv += 1
        vl /= max(nbv, 1)

        log_event(jsonl, "epoch", {"epoch": ep, "train_gll": tr_loss, "val_gll": vl})
        if mlrun:
            try_mlflow_log_metrics({"train_gll": tr_loss, "val_gll": vl}, step=ep)
        if wandb_run:
            try:
                import wandb

                wandb.log({"train_gll": tr_loss, "val_gll": vl, "epoch": ep})
            except Exception:
                pass

        if vl < best_val:
            best_val = vl
            pth = os.path.join(run_dir, "model_best.pt")
            torch.save(
                {"model": model.state_dict(), "config": config, "epoch": ep}, pth
            )
            log_event(jsonl, "checkpoint", {"epoch": ep, "path": pth, "val_gll": vl})

    log_event(jsonl, "end.train", {"run_dir": run_dir, "best_val_gll": best_val})
    return {"run_dir": run_dir, "best_val_gll": best_val}
