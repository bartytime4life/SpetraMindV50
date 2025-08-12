from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# Model factory
from src.spectramind.models import build_model


# ---------------------------
# Small utilities
# ---------------------------

def gll_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Gaussian negative log-likelihood per (B, N) with sigma>0.
    Loss = 0.5 * [ log(2πσ^2) + (y - μ)^2 / σ^2 ]
    """
    eps = 1e-8
    sigma2 = sigma.clamp_min(eps) ** 2
    return 0.5 * (torch.log(2 * math.pi * sigma2) + (target - mu) ** 2 / sigma2).mean()


def maybe_symbolic_penalty(mu: torch.Tensor) -> torch.Tensor:
    """
    Placeholder symbolic regularizer: encourages smoothness across spectral bins
    via L2 of second difference. BxN -> scalar
    You can swap this with your real rules engine later.
    """
    # Finite-difference along spectral axis
    d1 = mu[:, 1:] - mu[:, :-1]                   # (B, N-1)
    d2 = d1[:, 1:] - d1[:, :-1]                   # (B, N-2)
    return (d2**2).mean()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------
# Data
# ---------------------------

@dataclass
class Batch:
    fgs1: torch.Tensor           # (B, T, F_fgs1)
    airs: torch.Tensor           # (B, N, F_airs)
    edges: torch.Tensor | None   # (B, N, N, E) or None
    y_mu: torch.Tensor | None    # (B, N) or None
    y_sigma: torch.Tensor | None # (B, N) or None


class TinySyntheticDataset(Dataset):
    """
    Minimal dataset so training loop runs before you wire real data.
    Shapes are consistent with the model scaffold: FGS1 sequence and AIRS graph.
    """
    def __init__(self, n_samples=64, T=64, F_fgs1=4, N=283, F_airs=256, E=4, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.fgs1 = torch.randn(n_samples, T, F_fgs1, generator=g)
        self.airs = torch.randn(n_samples, N, F_airs, generator=g)
        self.edges = torch.randn(n_samples, N, N, E, generator=g)
        # synthetic "truth"
        base = torch.tanh(self.airs.mean(dim=-1))  # (S, N)
        self.y_mu = base + 0.05 * torch.randn_like(base, generator=g)
        self.y_sigma = 0.1 + 0.02 * torch.rand_like(base, generator=g)

    def __len__(self) -> int:
        return self.fgs1.size(0)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return dict(
            fgs1=self.fgs1[idx],
            airs=self.airs[idx],
            edges=self.edges[idx],
            y_mu=self.y_mu[idx],
            y_sigma=self.y_sigma[idx],
        )


def load_jsonl_list(path: str | Path) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def make_loader(
    split_path: str | Path | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """
    Replace this stub with your real dataset that reads split files and loads tensors.
    For now, if split file does not exist, we yield TinySyntheticDataset.
    """
    if split_path and Path(split_path).exists():
        # TODO: implement your real dataset loader here
        ds = TinySyntheticDataset()
    else:
        ds = TinySyntheticDataset()
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


# ---------------------------
# MAE Head (for pretrain)
# ---------------------------

class MAEHead(nn.Module):
    """
    Lightweight reconstruction head for AIRS node embeddings.
    Projects from latent D -> original AIRS feature dim (F_airs).
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) -> (B, N, F_airs)
        return self.net(x)


def apply_airs_mask(x: torch.Tensor, mask_ratio: float, generator: torch.Generator | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask a fraction of AIRS nodes by zeroing features.
    Returns (x_masked, mask_bool) where mask_bool is True for masked nodes.
    """
    B, N, F = x.shape
    num_mask = int(mask_ratio * N)
    if num_mask <= 0:
        return x, torch.zeros(B, N, dtype=torch.bool, device=x.device)

    mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
    for b in range(B):
        idx = torch.randperm(N, generator=generator, device=x.device)[:num_mask]
        mask[b, idx] = True
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


# ---------------------------
# Optimizers / schedulers
# ---------------------------

def build_optimizer(name: str, params, cfg_optim: DictConfig) -> torch.optim.Optimizer:
    name = (name or "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=float(cfg_optim.lr), betas=tuple(cfg_optim.betas), eps=float(cfg_optim.eps), weight_decay=float(cfg_optim.weight_decay))
    elif name == "sgd":
        return torch.optim.SGD(params, lr=float(cfg_optim.lr), momentum=float(cfg_optim.momentum), weight_decay=float(cfg_optim.weight_decay), nesterov=bool(cfg_optim.nesterov))
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(name: str, optim: torch.optim.Optimizer, cfg_sched: DictConfig):
    name = (name or "cosine").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(cfg_sched.T_max), eta_min=float(cfg_sched.eta_min))
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optim, step_size=int(cfg_sched.step_size), gamma=float(cfg_sched.gamma))
    else:
        raise ValueError(f"Unknown scheduler: {name}")


# ---------------------------
# Trainer
# ---------------------------

@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val: float = float("inf")


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if (cfg.train.device == "cuda" and torch.cuda.is_available()) else "cpu")

        # Build model
        self.model = build_model(cfg)
        self.model.to(self.device)

        # Optional: MAE head (initialized lazily when we know F_airs)
        self.mae_head: nn.Module | None = None

        # Data
        self.train_loader = make_loader(cfg.train.train_split, cfg.train.batch_size, cfg.train.shuffle, cfg.train.num_workers)
        self.val_loader = make_loader(cfg.train.val_split, cfg.train.val_batch_size, False, cfg.train.num_workers)

        # Reg/grad
        self.clip_grad = float(cfg.train.regularization.clip_grad_norm)

        # Checkpoints
        self.save_dir = Path(cfg.train.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # AMP
        self.use_amp = bool(cfg.train.amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    # -------- Phase runners --------

    def run_mae_phase(self, epochs: int, mask_ratio: float, cfg_phase: DictConfig):
        self.model.train()
        # Build a small MAE head on first batch when we learn F_airs
        opt = build_optimizer(cfg_phase.optimizer, list(self.model.parameters()), self.cfg.train.optimizer)
        sch = build_scheduler(cfg_phase.scheduler, opt, self.cfg.train.scheduler)

        for ep in range(epochs):
            for batch in self.train_loader:
                fgs1 = batch["fgs1"].to(self.device)                  # (B, T, F_fgs1)
                airs = batch["airs"].to(self.device)                  # (B, N, F_airs)
                edges = batch.get("edges")
                if edges is not None:
                    edges = edges.to(self.device)

                if self.mae_head is None:
                    F_airs = airs.size(-1)
                    # Take the AIRS encoder from the model via forward with zero grads
                    # We will use the fused representation to reconstruct original AIRS feats.
                    self.mae_head = MAEHead(in_dim=self.model.enc_airs.layers[0].q.in_features // self.model.enc_airs.n_heads, out_dim=F_airs)
                    # The above may not match perfectly; simpler: project from the encoder output D
                    self.mae_head = MAEHead(in_dim=self.model.enc_airs.in_proj.out_features, out_dim=F_airs).to(self.device)

                airs_masked, mask_bool = apply_airs_mask(airs, mask_ratio, generator=torch.Generator(device=airs.device))
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Encode AIRS only; ignore FGS1 here (pure reconstruction pretrain)
                    h_nodes = self.model.enc_airs(airs_masked, edges)          # (B, N, D)
                    pred = self.mae_head(h_nodes)                              # (B, N, F_airs)
                    # MSE only on masked nodes
                    diff = (pred - airs)
                    if mask_bool.any():
                        loss = (diff[mask_bool] ** 2).mean()
                    else:
                        loss = (diff ** 2).mean()

                opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self.clip_grad and self.clip_grad > 0:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(opt)
                self.scaler.update()
                sch.step()

            self._save_ckpt(phase="mae", epoch=ep + 1)

    def run_supervised_phase(self, epochs: int, cfg_phase: DictConfig):
        opt = build_optimizer(cfg_phase.optimizer, self.model.parameters(), self.cfg.train.optimizer)
        sch = build_scheduler(cfg_phase.scheduler, opt, self.cfg.train.scheduler)

        for ep in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                fgs1 = batch["fgs1"].to(self.device)
                airs = batch["airs"].to(self.device)
                edges = batch.get("edges")
                edges = edges.to(self.device) if edges is not None else None
                # Targets: gracefully fallback to zeros if missing
                y_mu = batch.get("y_mu")
                y_sigma = batch.get("y_sigma")
                if y_mu is None:
                    y_mu = torch.zeros(airs.shape[:-1], device=self.device)
                else:
                    y_mu = y_mu.to(self.device)
                if y_sigma is None:
                    y_sigma = 0.1 * torch.ones_like(y_mu, device=self.device)
                else:
                    y_sigma = y_sigma.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out = self.model({"fgs1": fgs1, "airs": airs, "edges": edges})
                    mu, sigma = out["mu"], out["sigma"]
                    loss_gll = gll_loss(mu, sigma, y_mu)
                    loss_sym = maybe_symbolic_penalty(mu)
                    loss = loss_gll + float(self.cfg.model.symbolic.get("lambda_sm", 0.1)) * loss_sym

                opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self.clip_grad and self.clip_grad > 0:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(opt)
                self.scaler.update()
                sch.step()

            self._save_ckpt(phase="supervised", epoch=ep + 1)

    # -------- Orchestration --------

    def run(self):
        """
        Dispatch phases from cfg.train.curriculum.* while keeping things simple.
        """
        cur = self.cfg.train.curriculum
        phases = cur.get("phases", [])
        if not phases:
            # Fallback: single supervised if curriculum missing
            phases = [dict(name="supervised", epochs=1, loss="gll_symbolic_loss", optimizer="adamw", scheduler="cosine")]

        print(f"[Trainer] Model params: {count_params(self.model):,}")
        for p in phases:
            name = str(p["name"]).lower()
            print(f"[Trainer] Phase: {name}")
            if name == "mae":
                epochs = int(p.get("epochs", 1))
                mask_ratio = float(p.get("mask_ratio", 0.4))
                self.run_mae_phase(epochs=epochs, mask_ratio=mask_ratio, cfg_phase=OmegaConf.create(p))
            elif name == "supervised":
                epochs = int(p.get("epochs", 1))
                self.run_supervised_phase(epochs=epochs, cfg_phase=OmegaConf.create(p))
            else:
                print(f"[Trainer] Unknown phase '{name}', skipping.")

    # -------- Checkpointing --------

    def _save_ckpt(self, phase: str, epoch: int):
        ckpt = dict(
            model=self.model.state_dict(),
            epoch=epoch,
            phase=phase,
            cfg=OmegaConf.to_container(self.cfg, resolve=True),
        )
        path = self.save_dir / f"{phase}-epoch{epoch}.ckpt"
        torch.save(ckpt, path)
        print(f"[Trainer] Saved checkpoint: {path}")


# ---------------------------
# Public API
# ---------------------------

def train(cfg: DictConfig) -> None:
    """
    Entry point called by `python -m spectramind train`.
    """
    # Ensure sections exist (Hydra composition parity)
    # cfg.train.*, cfg.model.* expected by this trainer
    trainer = Trainer(cfg)
    trainer.run()