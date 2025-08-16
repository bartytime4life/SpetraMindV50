"""
Fusion of FFT PCA with symbolic fingerprints.
"""
import numpy as np
import matplotlib.pyplot as plt

def generate_fft_symbolic_fusion(mu, symbolic_vector, planet_id, outdir="diagnostics"):
    fusion = np.fft.rfft(mu)[:len(symbolic_vector)].real * symbolic_vector
    plt.plot(fusion)
    plt.title(f"FFT × Symbolic Fusion - {planet_id}")
    plt.savefig(f"{outdir}/fft_symbolic_fusion_{planet_id}.png")
    plt.close()
    return fusion
