"""
FFT and autocorrelation diagnostics for μ spectra.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_mu_fft(mu, planet_id, outdir="diagnostics"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fft_vals = np.fft.rfft(mu)
    power = np.abs(fft_vals) ** 2
    plt.plot(power)
    plt.title(f"FFT Power Spectrum - {planet_id}")
    plt.savefig(f"{outdir}/fft_power_{planet_id}.png")
    plt.close()
    return power
