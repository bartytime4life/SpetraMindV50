import argparse
import logging
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._config import DiagnosticsConfig, load_config_from_yaml_or_defaults
from ._io import ensure_dir, load_preds_truth, maybe_load_bin_meta, save_csv, save_json
from ._logging import capture_env_and_git, get_logger, log_event


def second_derivative_l2(mu: np.ndarray) -> np.ndarray:
    """Discrete second derivative magnitude per bin."""
    d2 = np.zeros_like(mu)
    d2[:, 1:-1] = np.abs(mu[:, :-2] - 2.0 * mu[:, 1:-1] + mu[:, 2:])
    d2[:, 0] = d2[:, 1]
    d2[:, -1] = d2[:, -2]
    return d2


def plot_smoothness_map(sm: np.ndarray, out_png: str, dpi: int = 120) -> None:
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.figure(figsize=(12, 6), dpi=dpi)
    im = plt.imshow(sm, aspect="auto", interpolation="nearest", cmap="magma")
    plt.colorbar(im, label="|∂²μ|")
    plt.xlabel("Bin index")
    plt.ylabel("Object index")
    plt.title("Spectral Smoothness Map (second derivative magnitude)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def summarize_by_region(sm: np.ndarray, meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    per_bin = sm.mean(axis=0)
    df = pd.DataFrame({"bin": np.arange(sm.shape[1]), "smoothness": per_bin})
    if meta is None:
        return df
    merge_cols = [c for c in ["molecule", "region", "instrument"] if c in meta.columns]
    dfm = df.merge(meta, on="bin", how="left")
    grp_cols = [c for c in merge_cols if c in dfm.columns]
    if not grp_cols:
        return dfm
    agg = dfm.groupby(grp_cols, dropna=False)["smoothness"].mean().reset_index()
    return agg


def run(
    cfg: DiagnosticsConfig,
    run_name: str = "spectral-smoothness-map",
) -> dict:
    logger = get_logger(level=getattr(logging, cfg.log.level.upper(), logging.INFO))
    capture_env_and_git(cfg.log.log_dir, logger=logger)
    log_event(
        "smoothness-start",
        {"pred_dir": cfg.io.pred_dir},
        log_dir=cfg.log.log_dir,
        logger=logger,
    )

    mu, _, _ = load_preds_truth(
        cfg.io.pred_dir, cfg.io.y_true, cfg.io.mu_name, cfg.io.sigma_name
    )
    sm = second_derivative_l2(mu)

    out_dir = cfg.io.output_dir
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "smoothness_map.npy"), sm)
    plot_smoothness_map(sm, os.path.join(out_dir, "smoothness_map.png"), dpi=120)

    meta = maybe_load_bin_meta(cfg.io.bin_meta_csv)
    agg = summarize_by_region(sm, meta)
    csv_path = os.path.join(out_dir, "smoothness_summary.csv")
    save_csv(agg, csv_path)

    save_json(
        {"map_npy": "smoothness_map.npy", "summary_csv": "smoothness_summary.csv"},
        os.path.join(out_dir, "smoothness_summary.json"),
    )
    log_event(
        "smoothness-finish", {"csv": csv_path}, log_dir=cfg.log.log_dir, logger=logger
    )
    return {"csv": csv_path, "png": os.path.join(out_dir, "smoothness_map.png")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="SpectraMind V50 — Spectral Smoothness Map"
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config_from_yaml_or_defaults(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
