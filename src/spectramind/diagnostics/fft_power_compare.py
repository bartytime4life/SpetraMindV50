"""
FFT Power Compare
-----------------
Computes FFT power spectrum for μ spectra, compares clusters,
and exports diagnostic plots.
"""

import matplotlib.pyplot as plt
import numpy as np


def fft_power(mu, save_png="fft_power.png"):
    fft_vals = np.fft.rfft(mu)
    power = np.abs(fft_vals) ** 2
    plt.semilogy(power)
    plt.title("FFT Power Spectrum")
    plt.savefig(save_png)
    plt.close()
    return power
