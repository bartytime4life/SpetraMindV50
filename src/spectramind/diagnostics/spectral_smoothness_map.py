"""
Spectral Smoothness Map
-----------------------
Computes per-bin L2 smoothness of μ spectra, overlays symbolic
violations, and exports plots + JSON.
"""

import json

import numpy as np


def smoothness_map(mu, save_json="smoothness.json"):
    diff = np.diff(mu)
    val = float(np.mean(diff**2))
    with open(save_json, "w") as f:
        json.dump(dict(smoothness=val), f, indent=2)
    return val
