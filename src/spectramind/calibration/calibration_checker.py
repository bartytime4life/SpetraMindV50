"""Metrics and utilities for sigma calibration evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def evaluate_sigma_calibration(
    mu_pred: np.ndarray, mu_true: np.ndarray, sigma_pred: np.ndarray
) -> Dict[str, Any]:
    resid = mu_pred - mu_true
    z = resid / (sigma_pred + 1e-9)
    return {
        "coverage_68": float(np.mean(np.abs(z) <= 1.0)),
        "coverage_95": float(np.mean(np.abs(z) <= 2.0)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
    }


def save_summary_json(summary: Dict[str, Any], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
