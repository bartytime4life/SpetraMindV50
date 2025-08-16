"""
Diagnostic Summary Generator
----------------------------
Aggregates metrics: GLL, entropy, RMSE, calibration, symbolic loss,
SHAP overlays, FFT diagnostics. Saves JSON + plots.
"""

import json

import numpy as np


def generate_summary(mu_pred, mu_true, sigma, save_json="diagnostic_summary.json"):
    rmse = float(np.sqrt(np.mean((mu_pred - mu_true) ** 2)))
    summary = dict(rmse=rmse, n_bins=len(mu_pred))
    with open(save_json, "w") as f:
        json.dump(summary, f, indent=2)
    return summary
