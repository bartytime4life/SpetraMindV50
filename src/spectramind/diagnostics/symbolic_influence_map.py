import argparse
import logging
import os
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._config import DiagnosticsConfig, load_config_from_yaml_or_defaults
from ._io import ensure_dir, load_preds_truth, save_csv, save_json
from ._logging import capture_env_and_git, get_logger, log_event


def compute_symbolic_influence(
    mu: np.ndarray,
    violation_masks: np.ndarray,
    rule_weights: Optional[np.ndarray] = None,
    mode: str = "weighted_sum",
) -> np.ndarray:
    """Compute per-bin symbolic influence maps from violation masks."""
    n, b = mu.shape
    if (
        violation_masks.ndim != 3
        or violation_masks.shape[0] != n
        or violation_masks.shape[2] != b
    ):
        raise ValueError("violation_masks must be [N,R,B] aligned with mu [N,B].")
    r = violation_masks.shape[1]
    vm = violation_masks.astype(float)
    if rule_weights is not None:
        if rule_weights.shape[0] != r:
            raise ValueError("rule_weights must match number of rules R.")
        vm = vm * rule_weights[None, :, None]
    if mode == "max":
        influence = vm.max(axis=1)
    elif mode == "sum":
        influence = vm.sum(axis=1)
    else:
        influence = vm.sum(axis=1) / max(1, r)
    return influence


def render_influence_heatmap(
    infl: np.ndarray,
    out_png: str,
    dpi: int = 120,
    title: str = "Symbolic Influence Map",
) -> None:
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.figure(figsize=(12, 6), dpi=dpi)
    im = plt.imshow(infl, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(im, label="Influence")
    plt.xlabel("Bin index")
    plt.ylabel("Object index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def run(
    cfg: DiagnosticsConfig,
    violation_masks_path: str,
    rule_weights_path: Optional[str],
    mode: str = "weighted_sum",
    run_name: str = "symbolic-influence-map",
) -> Dict[str, str]:
    logger = get_logger(level=getattr(logging, cfg.log.level.upper(), logging.INFO))
    capture_env_and_git(cfg.log.log_dir, logger=logger)
    log_event(
        "symbolic-influence-start",
        {"pred_dir": cfg.io.pred_dir},
        log_dir=cfg.log.log_dir,
        logger=logger,
    )

    mu, _, _ = load_preds_truth(
        cfg.io.pred_dir, cfg.io.y_true, cfg.io.mu_name, cfg.io.sigma_name
    )
    vm = np.load(violation_masks_path)
    rule_weights = None
    if rule_weights_path and os.path.exists(rule_weights_path):
        rw = np.load(rule_weights_path)
        rule_weights = rw.reshape(-1)

    infl = compute_symbolic_influence(mu, vm, rule_weights, mode=mode)
    out_dir = cfg.io.output_dir
    npy_path = os.path.join(out_dir, "symbolic_influence_map.npy")
    png_path = os.path.join(out_dir, "symbolic_influence_map.png")
    csv_path = os.path.join(out_dir, "symbolic_influence_stats.csv")
    json_path = os.path.join(out_dir, "symbolic_influence_summary.json")

    ensure_dir(out_dir)
    np.save(npy_path, infl)
    render_influence_heatmap(infl, png_path, dpi=120)

    per_bin = infl.mean(axis=0)
    per_obj = infl.mean(axis=1)
    df = pd.DataFrame({"bin": np.arange(infl.shape[1]), "mean_influence": per_bin})
    save_csv(df, csv_path)
    save_json(
        {
            "per_bin_mean": per_bin.tolist(),
            "per_obj_mean": per_obj.tolist(),
            "mode": mode,
            "violation_masks": os.path.basename(violation_masks_path),
            "rule_weights": (
                os.path.basename(rule_weights_path) if rule_weights_path else None
            ),
        },
        json_path,
    )

    log_event(
        "symbolic-influence-finish",
        {"npy": npy_path, "png": png_path, "csv": csv_path},
        log_dir=cfg.log.log_dir,
        logger=logger,
    )
    return {"npy": npy_path, "png": png_path, "csv": csv_path, "json": json_path}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="SpectraMind V50 — Symbolic Influence Map"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--violation-masks", type=str, required=True)
    parser.add_argument("--rule-weights", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="weighted_sum",
        choices=["weighted_sum", "sum", "max"],
    )
    args = parser.parse_args(argv)
    cfg = load_config_from_yaml_or_defaults(args.config)
    run(cfg, args.violation_masks, args.rule_weights, mode=args.mode)


if __name__ == "__main__":
    main()
