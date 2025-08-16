from __future__ import annotations

"""
Self-test for src/spectramind/train package.

This script validates:

* Importability of modules
* Presence of essential symbols
* Ability to build dummy schedulers/loggers
* File-system write access for logs and artifacts
"""

from pathlib import Path
from typing import Dict

from . import callbacks, common, data_loading, logger, losses, schedulers


def run_selftest() -> Dict[str, str]:
    # Create dirs
    common.make_dirs("logs", "artifacts")
    # Logger
    lg = logger.get_logger("spectramind.train.selftest")
    lg.info("Logger OK")
    # JSONL
    logger.write_jsonl({"event": "selftest", "ok": True})
    # Scheduler dummy
    schedulers.cosine_with_warmup(optimizer=None, warmup_steps=10, total_steps=100)  # type: ignore
    # Loss numeric fallback
    gll = losses.gaussian_log_likelihood(0.0, 1.0, 0.0)  # type: ignore
    # Checkpoint write
    ckpt = common.save_checkpoint(
        {"hello": "world"}, ckpt_dir="artifacts/_selftest", tag="t0", keep_last=2
    )
    assert Path(ckpt).exists()
    # Manifest
    common.export_manifest(
        {"ckpt": ckpt, "gll_scalar": float(gll)}, "artifacts/_selftest_manifest.json"
    )
    return {"status": "ok", "ckpt": ckpt}


if __name__ == "__main__":  # pragma: no cover
    print(run_selftest())

