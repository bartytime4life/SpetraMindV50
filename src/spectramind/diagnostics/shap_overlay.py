"""
SHAP Overlay for SpectraMind V50:
- SHAP × μ spectra
- Top-K bins visualization
- JSON metadata export
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def overlay_shap(mu, shap_values, planet_id: str, outdir="diagnostics"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.plot(mu, label="μ Spectrum")
    plt.bar(range(len(shap_values)), shap_values, alpha=0.5, label="SHAP")
    plt.legend()
    plt.title(f"SHAP Overlay - {planet_id}")
    plt.savefig(f"{outdir}/shap_overlay_{planet_id}.png")
    plt.close()
    summary = {"planet_id": planet_id, "mean_shap": float(np.mean(shap_values))}
    with open(f"{outdir}/shap_overlay_{planet_id}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
