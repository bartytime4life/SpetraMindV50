# -*- coding: utf-8 -*-
"""SpectraMind V50 — Submission Packaging"""
from __future__ import annotations

import datetime
import json
import zipfile
from typing import Any, Dict, Optional

import pathlib

from .utils_infer import capture_git_state, capture_python_env, compute_config_hash, write_json


def build_manifest(
    run_dir: pathlib.Path,
    cfg_hash: str,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = datetime.datetime.now().isoformat()
    manifest = {
        "created": now,
        "config_hash": cfg_hash,
        "git": capture_git_state(),
        "env": capture_python_env(),
        "artifacts": sorted(
            [str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()]
        ),
    }
    if extras:
        manifest.update(extras)
    write_json(run_dir / "artifacts" / "manifest.json", manifest)
    return manifest


def make_zip_bundle(run_dir: pathlib.Path, out_name: Optional[str] = None) -> pathlib.Path:
    out_name = out_name or f"submission_bundle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = run_dir / out_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        sub = run_dir / "submission" / "submission.csv"
        if sub.exists():
            z.write(sub, arcname=f"submission/{sub.name}")
        for rel in [
            "artifacts/calibration_summary.json",
            "artifacts/diagnostics_summary.json",
            "artifacts/manifest.json",
        ]:
            p = run_dir / rel
            if p.exists():
                z.write(p, arcname=rel)
        for p in (run_dir / "artifacts").glob("*.html"):
            z.write(p, arcname=f"artifacts/{p.name}")
    return zip_path
