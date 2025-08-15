import numpy as np


def smoothness_penalty(mu: np.ndarray, lam: float = 1e-3) -> float:
    """Naive finite-difference L2 smoothness penalty."""
    d = np.diff(mu, axis=-1)
    return float(lam * (d**2).mean())
