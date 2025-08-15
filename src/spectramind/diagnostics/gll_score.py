import numpy as np


def gll(mu_pred: np.ndarray, sigma_pred: np.ndarray, mu_true: np.ndarray) -> float:
    var = np.square(sigma_pred) + 1e-12
    return float(
        0.5 * (np.log(2 * np.pi * var) + np.square(mu_true - mu_pred) / var).mean()
    )
