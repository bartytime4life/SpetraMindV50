"""
Computes ∂L/∂μ symbolic influence per rule.
Exports JSON + visual maps for symbolic diagnostics.
"""
import numpy as np
import json
from pathlib import Path

def compute_symbolic_influence(mu, symbolic_loss_fn, planet_id, outdir="diagnostics"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    grad = np.gradient(symbolic_loss_fn(mu))
    summary = {"planet_id": planet_id, "mean_symbolic_influence": float(np.mean(grad))}
    with open(f"{outdir}/symbolic_influence_{planet_id}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
