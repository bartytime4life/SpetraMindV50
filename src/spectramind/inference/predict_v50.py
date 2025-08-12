from __future__ import annotations

import csv
from pathlib import Path

NUM_BINS = 283


def predict(out_csv: Path) -> None:
    # Write a minimal valid submission with constant μ/σ (for CI wire-up)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [f"mu_{i}" for i in range(NUM_BINS)] + [f"sigma_{i}" for i in range(NUM_BINS)]
        w.writerow(header)
        # single dummy row
        w.writerow([0.0] * NUM_BINS + [0.1] * NUM_BINS)
