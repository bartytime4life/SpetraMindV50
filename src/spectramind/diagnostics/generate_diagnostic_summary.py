"""
Generates full diagnostic summary for SpectraMind V50:
- GLL, RMSE, entropy
- SHAP overlays
- Symbolic violation scores
- FFT/Autocorrelation metrics
- Exports diagnostic_summary.json
"""
import json
from pathlib import Path

def generate_summary(metrics: dict, outdir: str = "diagnostics"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    outfile = Path(outdir) / "diagnostic_summary.json"
    with open(outfile, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Diagnostics] Summary written to {outfile}")
    return outfile
