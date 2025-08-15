import numpy as np


def temperature_scale_sigma(sigma: np.ndarray, T: float = 1.0) -> np.ndarray:
    return sigma * float(T)
