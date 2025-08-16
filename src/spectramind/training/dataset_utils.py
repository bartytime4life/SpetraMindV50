from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, RandomSampler

from .utils import import_from_string, seed_everything


def _seed_worker(worker_id: int) -> None:
    import random, numpy as np

    # Ensure each worker has a distinct but deterministic seed
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders from config.
    Expect:
    cfg["data"]["train_builder"] = "module.path:make_train_dataset"
    cfg["data"]["val_builder"]   = "module.path:make_val_dataset"
    cfg["data"]["batch_size"]    = int
    cfg["data"]["num_workers"]   = int
    cfg["seed"]                  = int
    """
    seed_everything(int(cfg.get("seed", 42)))

    data_cfg = dict(cfg.get("data", {}))
    bs = int(data_cfg.get("batch_size", 8))
    nw = int(data_cfg.get("num_workers", 4))
    pin = bool(data_cfg.get("pin_memory", True))
    drop_last = bool(data_cfg.get("drop_last", True))

    train_builder = data_cfg.get("train_builder", None)
    val_builder = data_cfg.get("val_builder", None)
    if train_builder is None or val_builder is None:
        raise ValueError("data.train_builder and data.val_builder must be provided as 'module:func'")

    make_train = import_from_string(train_builder)
    make_val = import_from_string(val_builder)

    train_ds = make_train(data_cfg)
    val_ds = make_val(data_cfg)

    # Sampler: allow over-ride
    if data_cfg.get("sampler", "random").lower() == "random":
        train_sampler = RandomSampler(train_ds)
    else:
        train_sampler = None  # fallback to default if custom sampler created by dataset

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        shuffle=False if train_sampler is not None else True,
        num_workers=nw,
        pin_memory=pin,
        drop_last=drop_last,
        worker_init_fn=_seed_worker,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        drop_last=False,
        worker_init_fn=_seed_worker,
        persistent_workers=nw > 0,
    )
    logging.info(
        "Built dataloaders: train=%d samples, val=%d samples, batch_size=%d",
        len(train_ds),
        len(val_ds),
        bs,
    )
    return train_loader, val_loader
