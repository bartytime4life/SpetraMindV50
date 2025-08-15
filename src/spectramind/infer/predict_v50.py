# -*- coding: utf-8 -*-
"""SpectraMind V50 — Predict Entrypoint"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .calibrate_uncertainty_v50 import calibrate_predictions
from .ensemble_predict_v50 import aggregate_ensemble
from .selftest_infer import run_selftest
from .utils_infer import (
    InferenceConfig,
    append_to_debug_log,
    build_dataloader_from_config,
    build_model_from_config,
    compute_config_hash,
    ensure_run_dir,
    get_ids_from_batch,
    load_config,
    set_global_seed,
    setup_logging_stack,
    to_device,
    validate_submission_csv,
    write_json,
    write_jsonl_event,
    write_submission_csv,
)
from .package_submission_v50 import build_manifest, make_zip_bundle


def _inference_single_model(
    cfg: InferenceConfig,
    model: torch.nn.Module,
    device: torch.device,
    split: str = "test",
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    loader = build_dataloader_from_config(cfg, split=split)
    ids_all: List[str] = []
    mu_all: List[torch.Tensor] = []
    sigma_all: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device=device)
            mu, sigma = model(batch)
            sigma = torch.clamp(sigma, min=1e-6)
            ids = get_ids_from_batch(batch)
            ids_all.extend(ids)
            mu_all.append(mu.detach().cpu())
            sigma_all.append(sigma.detach().cpu())

    mu_full = torch.cat(mu_all, dim=0)
    sigma_full = torch.cat(sigma_all, dim=0)
    return ids_all, mu_full, sigma_full

def predict(
    cfg_path: Optional[str],
    overrides: Optional[Dict[str, Any]] = None,
    output_root: Optional[str] = None,
    tag: Optional[str] = None,
    selftest: bool = False,
    deep_selftest: bool = False,
    do_package: bool = True,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path, overrides)
    cfg_hash = compute_config_hash(cfg)
    run_dir = ensure_run_dir(base=output_root, tag=tag or "predict_v50")
    logs = setup_logging_stack(run_dir)
    logger = logging.getLogger("spectramind.infer")

    append_to_debug_log(pathlib.Path(logs["audit"]), f"- predict_v50 start | cfg_hash={cfg_hash} | tag={tag or ''}")

    if selftest:
        _ = run_selftest(cfg, run_dir=run_dir, deep=deep_selftest)

    set_global_seed(cfg.seed)
    device = torch.device(
        cfg.device if (torch.cuda.is_available() or "cpu" not in cfg.device) else "cpu"
    )
    logger.info("Using device: %s", device)

    model_entries: List[Dict[str, Any]] = []
    ckpt = cfg.model.get("ckpt")
    if isinstance(ckpt, list):
        for i, c in enumerate(ckpt):
            sub_cfg = InferenceConfig(**{**cfg.__dict__})
            sub_cfg.model = {**cfg.model, "ckpt": c}
            model_entries.append({"idx": i, "cfg": sub_cfg})
    else:
        model_entries.append({"idx": 0, "cfg": cfg})

    ensemble_preds: List[Tuple[List[str], torch.Tensor, torch.Tensor]] = []
    for entry in model_entries:
        mcfg: InferenceConfig = entry["cfg"]
        model = build_model_from_config(mcfg, device=device)
        ids, mu, sigma = _inference_single_model(mcfg, model, device=device, split="test")
        write_jsonl_event(pathlib.Path(logs["events"]), {"event": "model_infer_done", "model_idx": entry["idx"], "N": len(ids)})
        ensemble_preds.append((ids, mu, sigma))

    ids = ensemble_preds[0][0]
    for i, (ids_i, _, _) in enumerate(ensemble_preds):
        if ids_i != ids:
            raise RuntimeError(f"ID mismatch across ensemble member {i}")

    if len(ensemble_preds) == 1:
        mu, sigma = ensemble_preds[0][1], ensemble_preds[0][2]
        agg_summary = {"K": 1, "method_mu": "na", "method_sigma": "na"}
    else:
        K = len(ensemble_preds)
        mus = [p[1] for p in ensemble_preds]
        sigmas = [p[2] for p in ensemble_preds]
        mu, sigma, agg_summary = aggregate_ensemble(
            preds=list(zip(mus, sigmas)),
            method_mu=cfg.model.get("ensemble", {}).get("mu", "mean"),
            method_sigma=cfg.model.get("ensemble", {}).get("sigma", "geom_mean"),
            weights=None,
        )
        write_json(pathlib.Path(run_dir) / "artifacts" / "ensemble_summary.json", agg_summary)

    calib_target_path = cfg.calibration.get("target_path")
    calib = None
    if calib_target_path and os.path.exists(calib_target_path):
        target_np = np.load(calib_target_path)
        if target_np.shape == mu.shape:
            calib = {"target": torch.from_numpy(target_np).to(mu.dtype)}
        else:
            logging.getLogger("spectramind.infer").warning(
                "Calibration target shape mismatch; skipping calibration."
            )
    mu_cal, sigma_cal, cal_summary = calibrate_predictions(
        cfg, mu=mu, sigma=sigma, calib=calib, out_dir=pathlib.Path(run_dir) / "artifacts"
    )

    out_csv = pathlib.Path(run_dir) / "submission" / "submission.csv"
    fmt = cfg.submission.get("format", "wide")
    write_submission_csv(out_csv, ids=ids, mu=mu_cal, sigma=sigma_cal, fmt=fmt)
    val_summary = validate_submission_csv(out_csv, fmt=fmt, bins=cfg.bins, expect_ids=ids)
    write_json(pathlib.Path(run_dir) / "artifacts" / "submission_validation.json", val_summary)
    logging.getLogger("spectramind.infer").info("Submission written: %s", out_csv)

    manifest = build_manifest(
        run_dir,
        cfg_hash=cfg_hash,
        extras={"ensemble": agg_summary, "calibration": cal_summary},
    )
    zip_path = make_zip_bundle(run_dir) if do_package else None

    summary = {
        "cfg_hash": cfg_hash,
        "ids": len(ids),
        "bins": cfg.bins,
        "submission": str(out_csv),
        "bundle": str(zip_path) if zip_path else None,
        "run_dir": str(run_dir),
        "device": str(device),
        "ensemble": agg_summary,
        "calibration": cal_summary,
        "validation": val_summary,
    }
    write_json(pathlib.Path(run_dir) / "artifacts" / "predict_summary.json", summary)
    append_to_debug_log(
        pathlib.Path(logs["audit"]),
        f"- predict_v50 summary: {json.dumps(summary)}",
    )
    write_jsonl_event(pathlib.Path(logs["events"]), {"event": "predict_complete", **summary})

    return summary

def _parse_kv_overrides(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in kvs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if v.lower() in ("true", "false"):
            vv: Any = v.lower() == "true"
        else:
            try:
                vv = int(v)
            except ValueError:
                try:
                    vv = float(v)
                except ValueError:
                    vv = v
        out[k] = vv
    return out


def main() -> None:
    p = argparse.ArgumentParser("spectramind-v50-predict")
    p.add_argument("--config", type=str, required=False, help="Path to YAML/Hydra config (config_v50.yaml).")
    p.add_argument("--output", type=str, required=False, default=None, help="Base output directory for run.")
    p.add_argument("--tag", type=str, required=False, default=None, help="Optional run tag suffix.")
    p.add_argument("--override", type=str, nargs="*", default=[], help="Override config values key=value (dotted keys allowed).")
    p.add_argument("--selftest", action="store_true", help="Run selftest before inference.")
    p.add_argument("--deep-selftest", action="store_true", help="Run deep selftest.")
    p.add_argument("--no-package", action="store_true", help="Do not create zip bundle.")
    args = p.parse_args()

    overrides = _parse_kv_overrides(args.override)
    summary = predict(
        cfg_path=args.config,
        overrides=overrides,
        output_root=args.output,
        tag=args.tag,
        selftest=args.selftest,
        deep_selftest=args.deep_selftest,
        do_package=not args.no_package,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
