import argparse
import logging
import os
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._config import DiagnosticsConfig, load_config_from_yaml_or_defaults
from ._io import ensure_dir, load_preds_truth, save_csv, save_json, try_load
from ._logging import capture_env_and_git, get_logger, log_event


def compute_pseudo_shap(
    mu: np.ndarray, y: Optional[np.ndarray], sigma: Optional[np.ndarray]
) -> np.ndarray:
    """Fallback pseudo-SHAP when true SHAP attributions are not provided."""
    if y is None or sigma is None:
        centered = mu - np.nanmean(mu, axis=1, keepdims=True)
        return centered / (np.std(centered, axis=1, keepdims=True) + 1e-6)
    inv_var = 1.0 / (np.maximum(sigma, 1e-6) ** 2)
    shap_like = (mu - y) * inv_var
    shap_like = shap_like / (np.std(shap_like, axis=1, keepdims=True) + 1e-6)
    return shap_like


def overlay_topk_bins(
    mu: np.ndarray, shap_vals: np.ndarray, k: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """For each object, select top-K |SHAP| bins and return masks and indices."""
    abs_s = np.abs(shap_vals)
    idx = np.argpartition(-abs_s, kth=min(k, abs_s.shape[1] - 1), axis=1)[:, :k]
    mask = np.zeros_like(mu, dtype=bool)
    rows = np.arange(mu.shape[0])[:, None]
    mask[rows, idx] = True
    return mask, idx


def plot_overlay(
    mu: np.ndarray,
    shap_vals: np.ndarray,
    mask: np.ndarray,
    out_png: str,
    dpi: int = 120,
) -> None:
    """Create a compact overlay summary."""
    ensure_dir(os.path.dirname(out_png) or ".")
    mean_mu = mu.mean(axis=0)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    topk_mask_fraction = mask.mean(axis=0)

    fig, ax1 = plt.subplots(figsize=(12, 4), dpi=dpi)
    ax1.plot(mean_mu, label="mean μ")
    ax1.set_xlabel("Bin index")
    ax1.set_ylabel("μ (mean)")
    ax2 = ax1.twinx()
    ax2.plot(mean_abs_shap, alpha=0.7, label="mean |pseudo-SHAP|", linestyle="--")
    ax2.set_ylabel("|SHAP| (mean)")
    ax1.fill_between(
        np.arange(len(topk_mask_fraction)),
        0,
        topk_mask_fraction,
        color="grey",
        alpha=0.2,
        label="top-K fraction",
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("μ × pseudo-SHAP overlay (top-K coverage shaded)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def run(
    cfg: DiagnosticsConfig,
    shap_path: Optional[str],
    top_k: int = 20,
    run_name: str = "shap-overlay",
) -> dict:
    logger = get_logger(level=getattr(logging, cfg.log.level.upper(), logging.INFO))
    capture_env_and_git(cfg.log.log_dir, logger=logger)
    log_event(
        "shap-overlay-start",
        {"pred_dir": cfg.io.pred_dir},
        log_dir=cfg.log.log_dir,
        logger=logger,
    )

    mu, sigma, y_true = load_preds_truth(
        cfg.io.pred_dir, cfg.io.y_true, cfg.io.mu_name, cfg.io.sigma_name
    )
    shap_vals = try_load(shap_path)
    if shap_vals is None:
        shap_vals = compute_pseudo_shap(mu, y_true, sigma)
    if shap_vals.shape != mu.shape:
        raise ValueError(
            f"SHAP array must match μ shape. got {shap_vals.shape}, expected {mu.shape}"
        )

    mask, _ = overlay_topk_bins(mu, shap_vals, k=top_k)
    out_dir = cfg.io.output_dir
    ensure_dir(out_dir)

    np.save(os.path.join(out_dir, "pseudo_shap.npy"), shap_vals)
    np.save(os.path.join(out_dir, "topk_mask.npy"), mask)
    plot_overlay(
        mu, shap_vals, mask, os.path.join(out_dir, "shap_overlay.png"), dpi=120
    )

    df = pd.DataFrame(
        {
            "bin": np.arange(mu.shape[1]),
            "mean_mu": mu.mean(axis=0),
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
            "topk_fraction": mask.mean(axis=0),
        }
    )
    csv_path = os.path.join(out_dir, "shap_overlay_stats.csv")
    save_csv(df, csv_path)
    summary = {
        "top_k": top_k,
        "stats_csv": os.path.basename(csv_path),
        "overlay_png": "shap_overlay.png",
        "pseudo_shap": shap_path is None,
    }
    save_json(summary, os.path.join(out_dir, "shap_overlay_summary.json"))

    log_event("shap-overlay-finish", summary, log_dir=cfg.log.log_dir, logger=logger)
    return {"csv": csv_path, "png": os.path.join(out_dir, "shap_overlay.png")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="SpectraMind V50 — SHAP × μ Overlay")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--shap", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    cfg = load_config_from_yaml_or_defaults(args.config)
    run(cfg, args.shap, top_k=args.top_k)


if __name__ == "__main__":
    main()
