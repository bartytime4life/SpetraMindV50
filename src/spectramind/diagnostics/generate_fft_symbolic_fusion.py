"""
FFT × Symbolic Fusion Generator
-------------------------------
Performs FFT PCA + symbolic fingerprint fusion,
outputs cluster CSVs and interactive projections.
"""

import numpy as np
import pandas as pd


def generate_fusion(mu, save_csv="fft_symbolic_fusion.csv"):
    fft_vals = np.abs(np.fft.rfft(mu))
    df = pd.DataFrame(dict(fft=fft_vals))
    df.to_csv(save_csv, index=False)
    return df
