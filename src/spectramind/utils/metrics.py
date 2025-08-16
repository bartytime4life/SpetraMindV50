from typing import Tuple, Dict, Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _ensure_np(x):
    if np is None:
        raise ImportError("NumPy not installed")
    return x if isinstance(x, np.ndarray) else np.asarray(x)


def gll_loss(mu, sigma, y, eps: float = 1e-6) -> float:
    """Gaussian log-likelihood (negative) for scalar/vector inputs."""
    mu = _ensure_np(mu)
    sigma = np.maximum(_ensure_np(sigma), eps)
    y = _ensure_np(y)
    term = 0.5 * (np.log(2.0 * np.pi * (sigma ** 2)) + ((y - mu) ** 2) / (sigma ** 2))
    return float(np.mean(term))


def binwise_gll(mu, sigma, y, eps: float = 1e-6):
    mu = _ensure_np(mu)
    sigma = np.maximum(_ensure_np(sigma), eps)
    y = _ensure_np(y)
    term = 0.5 * (np.log(2.0 * np.pi * (sigma ** 2)) + ((y - mu) ** 2) / (sigma ** 2))
    return term  # per-bin vector


def rmse(pred, target) -> float:
    pred = _ensure_np(pred)
    target = _ensure_np(target)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred, target) -> float:
    pred = _ensure_np(pred)
    target = _ensure_np(target)
    return float(np.mean(np.abs(pred - target)))


def calibration_error(mu, sigma, y, eps: float = 1e-6) -> Dict[str, Any]:
    """Simple calibration check via z-scores."""
    mu = _ensure_np(mu)
    sigma = np.maximum(_ensure_np(sigma), eps)
    y = _ensure_np(y)
    z = (y - mu) / sigma
    mean = float(np.mean(z))
    std = float(np.std(z))
    return {"z_mean": mean, "z_std": std}


def zscore_histogram(mu, sigma, y, bins: int = 21, range: Tuple[float, float] = (-5, 5), eps: float = 1e-6):
    mu = _ensure_np(mu)
    sigma = np.maximum(_ensure_np(sigma), eps)
    y = _ensure_np(y)
    z = (y - mu) / sigma
    hist, edges = np.histogram(z, bins=bins, range=range)
    return hist, edges
