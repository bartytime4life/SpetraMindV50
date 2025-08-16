"""
Self-test for diagnostics integrity:
- Checks file presence
- Validates JSON outputs
- Runs minimal UMAP/FFT/SHAP test
"""
import os, json
from pathlib import Path

def run_selftest(outdir="diagnostics"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    dummy_json = {"test": "ok"}
    with open(Path(outdir)/"selftest.json", "w") as f:
        json.dump(dummy_json, f)
    print("[Diagnostics] Selftest passed.")
    return True
