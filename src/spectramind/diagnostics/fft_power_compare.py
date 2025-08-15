import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._config import DiagnosticsConfig, load_config_from_yaml_or_defaults
from ._io import ensure_dir, load_preds_truth, save_csv, save_json
from ._logging import capture_env_and_git, get_logger, log_event


def compute_fft_power(mu: np.ndarray) -> np.ndarray:
    """Compute normalized FFT power spectrum per object."""
    f = np.fft.rfft(mu, axis=1)
    power = f.real**2 + f.imag**2
    power = power / (power.sum(axis=1, keepdims=True) + 1e-12)
    left = power[:, 1:-1][:, ::-1] if power.shape[1] > 2 else power[:, :0]
    full = np.concatenate([power, left], axis=1)
    b = mu.shape[1]
    if full.shape[1] >= b:
        return full[:, :b]
    pad = np.zeros((mu.shape[0], b - full.shape[1]))
    return np.concatenate([full, pad], axis=1)


def pca_reduce(x: np.ndarray, n: int = 2) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    comps = xc @ vt[:n].T
    return comps


def kmeans(x: np.ndarray, k: int = 8, iters: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    centroids = x[rng.choice(n, size=k, replace=False)]
    for _ in range(iters):
        dist2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        lab = dist2.argmin(axis=1)
        for j in range(k):
            if (lab == j).any():
                centroids[j] = x[lab == j].mean(axis=0)
    return lab, centroids


def plot_fft_clusters(
    pts2d: np.ndarray, labels: np.ndarray, out_png: str, dpi: int = 120
) -> None:
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.figure(figsize=(7, 6), dpi=dpi)
    for lab in np.unique(labels):
        m = labels == lab
        plt.scatter(pts2d[m, 0], pts2d[m, 1], s=10, alpha=0.7, label=f"c{lab}")
    plt.legend(markerscale=2)
    plt.title("FFT Power PCA Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def run(
    cfg: DiagnosticsConfig,
    k: int = 8,
    run_name: str = "fft-power-compare",
) -> dict:
    logger = get_logger(level=getattr(logging, cfg.log.level.upper(), logging.INFO))
    capture_env_and_git(cfg.log.log_dir, logger=logger)
    log_event(
        "fft-compare-start",
        {"pred_dir": cfg.io.pred_dir, "k": k},
        log_dir=cfg.log.log_dir,
        logger=logger,
    )

    mu, _, _ = load_preds_truth(
        cfg.io.pred_dir, cfg.io.y_true, cfg.io.mu_name, cfg.io.sigma_name
    )
    power = compute_fft_power(mu)
    pts2d = pca_reduce(power, n=2)
    labels, _ = kmeans(pts2d, k=k)

    out_dir = cfg.io.output_dir
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "fft_power.npy"), power)
    np.save(os.path.join(out_dir, "fft_pca2.npy"), pts2d)
    np.save(os.path.join(out_dir, "fft_clusters.npy"), labels)

    plot_fft_clusters(pts2d, labels, os.path.join(out_dir, "fft_clusters.png"), dpi=120)

    df = pd.DataFrame(
        {
            "obj": np.arange(mu.shape[0]),
            "c": labels,
            "pc1": pts2d[:, 0],
            "pc2": pts2d[:, 1],
        }
    )
    csv_path = os.path.join(out_dir, "fft_cluster_assignments.csv")
    save_csv(df, csv_path)

    save_json(
        {"k": k, "assignments_csv": os.path.basename(csv_path)},
        os.path.join(out_dir, "fft_compare_summary.json"),
    )
    log_event(
        "fft-compare-finish", {"csv": csv_path}, log_dir=cfg.log.log_dir, logger=logger
    )
    return {"csv": csv_path, "png": os.path.join(out_dir, "fft_clusters.png")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="SpectraMind V50 — FFT Power Compare & Clustering"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args(argv)
    cfg = load_config_from_yaml_or_defaults(args.config)
    run(cfg, k=args.k)


if __name__ == "__main__":
    main()
