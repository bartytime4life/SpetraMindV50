"""
Rule-based symbolic violation predictor.
Computes per-rule violation scores and saves outputs.
"""
import numpy as np
import json
from pathlib import Path

class SymbolicViolationPredictor:
    def __init__(self, rules: list):
        self.rules = rules

    def evaluate(self, mu: np.ndarray):
        return {rule: float(np.mean(eval(rule))) for rule in self.rules}

    def save(self, mu, planet_id, outdir="diagnostics"):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        violations = self.evaluate(mu)
        with open(f"{outdir}/symbolic_violations_{planet_id}.json", "w") as f:
            json.dump(violations, f, indent=2)
        return violations
