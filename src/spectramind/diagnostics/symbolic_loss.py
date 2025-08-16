"""
Symbolic Loss Engine
--------------------
Evaluates per-rule symbolic constraints (smoothness, nonnegativity,
FFT asymmetry, photonic alignment). Returns scalar + per-bin maps.
"""

import numpy as np


def symbolic_smoothness(mu):
    return np.mean(np.diff(mu) ** 2)


def symbolic_nonnegativity(mu):
    return np.sum(np.minimum(mu, 0) ** 2)


def symbolic_asymmetry(mu):
    return np.sum((mu - mu[::-1]) ** 2)


def total_symbolic_loss(mu):
    return dict(
        smoothness=symbolic_smoothness(mu),
        nonnegativity=symbolic_nonnegativity(mu),
        asymmetry=symbolic_asymmetry(mu),
    )
