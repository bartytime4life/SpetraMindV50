from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DataLoader = object  # type: ignore
    Dataset = object  # type: ignore


@dataclass
class DLSpec:
    """DataLoader specification container for train/val/test splits."""

    train: Any
    val: Any
    test: Optional[Any] = None


def _default_collate(batch):
    """Minimal collate; replace with project-specific collate if needed."""
    if torch is None:
        return batch
    return torch.utils.data.default_collate(batch)  # type: ignore


def build_dataloaders(cfg: Dict[str, Any]) -> DLSpec:
    """Construct PyTorch DataLoaders from Hydra cfg.

    Expected cfg structure::

        cfg.data = {
          "module": "spectramind.data.ariel_dataset:build",  # factory returning (train, val, test)
          "batch_size": 16,
          "num_workers": 4,
          "pin_memory": True
        }
    """

    module = cfg["data"]["module"]
    mod_name, fn_name = module.split(":")
    mod = __import__(mod_name, fromlist=[fn_name])
    build_fn = getattr(mod, fn_name)
    datasets: Tuple[Dataset, Dataset, Optional[Dataset]] = build_fn(cfg)  # type: ignore

    if torch is None:
        # Return raw datasets in environments without torch
        return DLSpec(train=datasets[0], val=datasets[1], test=datasets[2])

    bs = int(cfg["data"].get("batch_size", 16))
    nw = int(cfg["data"].get("num_workers", 4))
    pin = bool(cfg["data"].get("pin_memory", True))

    train_loader = DataLoader(
        datasets[0],
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        collate_fn=_default_collate,
    )
    val_loader = DataLoader(
        datasets[1],
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        collate_fn=_default_collate,
    )
    test_loader = None
    if datasets[2] is not None:
        test_loader = DataLoader(
            datasets[2],
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            collate_fn=_default_collate,
        )

    return DLSpec(train=train_loader, val=val_loader, test=test_loader)

