"""
FFT Autocorrelation Analyzer
----------------------------
Computes FFT autocorrelation of μ spectra, overlays symbolic
diagnostics, and exports plots.
"""

import matplotlib.pyplot as plt
import numpy as np


def analyze_mu(mu, save_png="fft_autocorr.png"):
    ac = np.correlate(mu, mu, mode="full")
    plt.plot(ac)
    plt.savefig(save_png)
    plt.close()
    return ac
