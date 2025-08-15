# -*- coding: utf-8 -*-
"""SpectraMind V50 — Inference Self-Test"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict

import torch

from .utils_infer import (
    InferenceConfig,
    append_to_debug_log,
    build_dataloader_from_config,
    build_model_from_config,
    ensure_run_dir,
    get_ids_from_batch,
    set_global_seed,
    setup_logging_stack,
    to_device,
    validate_submission_csv,
    write_json,
    write_jsonl_event,
    write_submission_csv,
)


def run_selftest(cfg: InferenceConfig, run_dir: pathlib.Path, deep: bool = False) -> Dict[str, Any]:
    """Execute a series of small operations to confirm end-to-end viability."""
    logger = logging.getLogger("spectramind.infer")
    logs = setup_logging_stack(run_dir)
    append_to_debug_log(pathlib.Path(logs["audit"]), f"- selftest started (deep={deep})")

    set_global_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() or "cpu" not in cfg.device else "cpu"
    )

    loader = build_dataloader_from_config(cfg, split="test")
    it = iter(loader)
    batch = next(it)
    logger.info("Loaded 1 batch from dataset builder.")

    model = build_model_from_config(cfg, device)
    model.eval()
    with torch.no_grad():
        out = model(to_device(batch, device))
    assert isinstance(out, (tuple, list)) and len(out) == 2, "Model must return (mu, sigma)"
    mu, sigma = out
    assert mu.ndim == 2 and sigma.ndim == 2, "mu/sigma must be [B, bins]"
    assert mu.shape == sigma.shape, "mu and sigma shapes mismatch"
    assert mu.shape[1] == cfg.bins, f"Expected bins={cfg.bins}, got {mu.shape[1]}"
    logger.info("Model forward shape checks passed: %s", list(mu.shape))

    ids = get_ids_from_batch(batch)
    art_dir = pathlib.Path(run_dir) / "artifacts"
    sub_csv = art_dir / "sanity_submission.csv"
    fmt = cfg.submission.get("format", "wide")
    write_submission_csv(sub_csv, ids=ids, mu=mu, sigma=sigma, fmt=fmt)
    v = validate_submission_csv(sub_csv, fmt=fmt, bins=cfg.bins, expect_ids=ids)
    logger.info("Submission validator summary: %s", v)

    summary = {
        "batch_size": mu.shape[0],
        "bins": mu.shape[1],
        "submission": v,
        "device": str(device),
        "deep": deep,
    }
    write_json(pathlib.Path(run_dir) / "logs" / "selftest_summary.json", summary)
    append_to_debug_log(pathlib.Path(logs["audit"]), f"- selftest summary: {json.dumps(summary)}")
    write_jsonl_event(pathlib.Path(logs["events"]), {"event": "selftest_complete", **summary})
    return summary
