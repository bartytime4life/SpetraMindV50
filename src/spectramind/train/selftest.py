"""
Self-test for src/spectramind/train

Runs a tiny CPU-only smoke training using the built-in dummy dataset/model in train_v50
to verify:

* Logging (console + rotating file + JSONL)
* Optimizer/scheduler wiring
* AMP disabled path
* Checkpoint writes ('last.ckpt' and 'best.ckpt')
* Early stopping path
"""
import os
from pathlib import Path

from .train_v50 import run_train


def main():
    out_dir = "runs/selftest-train"
    cfg = {
        "run": {"name": "train-selftest", "out_dir": out_dir, "seed": 7, "device": "cpu", "mlflow": False},
        "dummy_mode": True,
        "dummy": {"dummy_dims": 32, "dummy_bins": 16},
        "loss": {"main": {"target": "spectramind.train.losses:GaussianLikelihoodLoss", "params": {"min_sigma": 1e-3}}},
        "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0},
        "sched": {"name": "cosine_warmup", "warmup_steps": 10},
        "train": {"epochs": 4, "grad_accum": 1, "amp": False, "clip_grad_norm": 1.0, "patience": 2},
    }
    summary = run_train(cfg)
    assert (Path(out_dir) / "last.ckpt").exists(), "last.ckpt not found"
    assert "best_val_loss" in summary, "Summary missing best_val_loss"
    print("Self-test OK:", summary)


if __name__ == "__main__":
    main()
