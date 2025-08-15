# -*- coding: utf-8 -*-
"""SpectraMind V50 — Inference Utilities"""
from __future__ import annotations

import abc
import csv
import datetime
import hashlib
import importlib
import inspect
import json
import logging
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

try:  # optional dependency
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    OmegaConf = None
    DictConfig = None


# ---------------------------
# Typed config surface
# ---------------------------


@dataclass
class InferenceConfig:
    """Minimal inference config schema used across infer submodules."""

    seed: int = 42
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True
    bins: int = 283
    data: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
    submission: Dict[str, Any] = field(default_factory=lambda: {"format": "wide"})
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


# ---------------------------
# Filesystem helpers
# ---------------------------

def ensure_dir(path: pathlib.Path, exist_ok: bool = True) -> pathlib.Path:
    """Create directory if not exists; return path."""
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path


def ensure_run_dir(base: Optional[str] = None, tag: Optional[str] = None) -> pathlib.Path:
    """Create a timestamped run directory for logs, artifacts, and submission."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = (tag or "infer").replace("/", "_").replace(" ", "_")
    base_dir = pathlib.Path(base or "runs") / f"{ts}_{safe_tag}"
    ensure_dir(base_dir)
    ensure_dir(base_dir / "logs")
    ensure_dir(base_dir / "artifacts")
    ensure_dir(base_dir / "submission")
    return base_dir


def path_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: pathlib.Path, obj: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True)


def read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl_event(jsonl_path: pathlib.Path, event: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", datetime.datetime.now().isoformat())
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def append_to_debug_log(md_path: pathlib.Path, text: str) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")

# ---------------------------
# Logging stack
# ---------------------------

def setup_logging_stack(
    run_dir: pathlib.Path,
    level: int = logging.INFO,
    name: str = "spectramind.infer",
) -> Dict[str, pathlib.Path]:
    """Configure a rotating file logger, a JSONL event stream, and an audit log."""
    log_dir = ensure_dir(run_dir / "logs")
    log_path = log_dir / "v50_infer.log"
    jsonl_path = log_dir / "events_infer.jsonl"
    audit_md = log_dir / "v50_debug_log.md"

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    from logging.handlers import RotatingFileHandler

    fh = RotatingFileHandler(
        log_path, mode="a", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(fh)

    write_jsonl_event(jsonl_path, {"event": "logging_initialized", "level": level})
    append_to_debug_log(
        audit_md,
        f"### Boot: {datetime.datetime.now().isoformat()} — logging initialized",
    )

    return {"log": log_path, "events": jsonl_path, "audit": audit_md}


# ---------------------------
# Config loading & hashing
# ---------------------------

def _dict_hash(d: Any) -> str:
    try:
        blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except TypeError:
        blob = str(d).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def compute_config_hash(cfg: InferenceConfig) -> str:
    structure = {
        "seed": cfg.seed,
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "bins": cfg.bins,
        "model": cfg.model,
        "data": cfg.data,
        "calibration": cfg.calibration,
        "submission": cfg.submission,
        "diagnostics": cfg.diagnostics,
    }
    return _dict_hash(structure)


def load_config(
    cfg_path: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None
) -> InferenceConfig:
    raw: Any = None
    if cfg_path and os.path.exists(cfg_path):
        if OmegaConf is not None:
            raw = OmegaConf.load(cfg_path)
            raw = OmegaConf.to_container(raw, resolve=True)
        else:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
    else:
        raw = {}

    if overrides:
        for k, v in overrides.items():
            _assign_dotted(raw, k, v)

    icfg = InferenceConfig(
        seed=raw.get("seed", 42),
        device=raw.get("device", "cuda"),
        batch_size=int(raw.get("batch_size", 8)),
        num_workers=int(raw.get("num_workers", 2)),
        pin_memory=bool(raw.get("pin_memory", True)),
        bins=int(_get_nested(raw, "bins", 283)),
        data=dict(raw.get("data", {})),
        model=dict(raw.get("model", {})),
        calibration=dict(raw.get("calibration", {})),
        submission=dict(raw.get("submission", {"format": "wide"})),
        diagnostics=dict(raw.get("diagnostics", {})),
        paths=dict(raw.get("paths", {})),
        run=dict(raw.get("run", {})),
        raw=raw,
    )
    return icfg


def _assign_dotted(dct: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = dct
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _get_nested(dct: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = dct
    for p in dotted.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur

# ---------------------------
# Git and environment capture
# ---------------------------

def capture_git_state(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort git state capture."""
    repo_root = repo_root or "."

    def _run(cmd: List[str]) -> str:
        try:
            out = subprocess.check_output(cmd, cwd=repo_root, stderr=subprocess.DEVNULL)
            return out.decode("utf-8").strip()
        except Exception:
            return ""

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
        "remote": _run(["git", "remote", "-v"]),
    }


def capture_python_env() -> Dict[str, Any]:
    """Capture minimal Python environment info."""
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }


# ---------------------------
# Model & dataset dynamic loading
# ---------------------------


class InferenceModel(abc.ABC, torch.nn.Module):
    """Minimal protocol for a model used in this pipeline."""

    @abc.abstractmethod
    def forward(self, batch: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, sigma)."""


def import_from_path(qualified: str) -> Any:
    """Import a symbol from a fully qualified path 'package.module:Class'."""
    if ":" in qualified:
        mod_name, sym_name = qualified.split(":", 1)
    else:
        parts = qualified.split(".")
        mod_name, sym_name = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(mod_name)
    return getattr(module, sym_name)


def build_model_from_config(cfg: InferenceConfig, device: torch.device) -> InferenceModel:
    model_def = cfg.model.get("registry") or cfg.model.get("class")
    model_kwargs = cfg.model.get("kwargs", {})
    ckpt = cfg.model.get("ckpt")
    if model_def:
        symbol = import_from_path(model_def)
        model = (
            symbol(**model_kwargs)
            if inspect.isclass(symbol)
            else symbol(cfg=cfg, **model_kwargs)
        )
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                f"Model registry/class did not return nn.Module: {type(model)}"
            )
        model = model.to(device)
        if ckpt and isinstance(ckpt, str) and ckpt.endswith(".pth"):
            _load_state_dict_safe(model, ckpt, device)
        return model  # type: ignore[return-value]
    if ckpt and isinstance(ckpt, str) and ckpt.endswith(".pt"):
        ts_model = torch.jit.load(ckpt, map_location=device)
        if not isinstance(ts_model, torch.nn.Module):
            raise ValueError("TorchScript object is not an nn.Module")
        return ts_model  # type: ignore[return-value]
    raise ValueError(
        "No valid model route found. Provide model.registry/class or TorchScript ckpt."
    )


def _load_state_dict_safe(
    model: torch.nn.Module, ckpt_path: str, device: torch.device
) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger = logging.getLogger("spectramind.infer")
    if missing:
        logger.warning("Missing keys in state_dict: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in state_dict: %s", unexpected)


def build_dataloader_from_config(
    cfg: InferenceConfig, split: str
) -> Iterable[Dict[str, Any]]:
    logger = logging.getLogger("spectramind.infer")
    bs = cfg.batch_size
    nw = cfg.num_workers
    pm = cfg.pin_memory

    split_over = (cfg.data.get("split_overrides") or {}).get(split, {})
    kwargs = dict(cfg.data.get("kwargs", {}))
    kwargs.update(split_over)

    if "registry" not in cfg.data:
        raise ValueError("cfg.data.registry not set (expected 'module:builder' path).")
    builder = import_from_path(cfg.data["registry"])
    dataset = builder(split=split, cfg=cfg, **kwargs)

    collate_fn = None
    if cfg.data.get("collate"):
        collate_fn = import_from_path(cfg.data["collate"])

    if cfg.data.get("loader"):
        loader_builder = import_from_path(cfg.data["loader"])
        return loader_builder(
            dataset=dataset,
            batch_size=bs,
            num_workers=nw,
            pin_memory=pm,
            collate_fn=collate_fn,
        )

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=bs,
        num_workers=nw,
        pin_memory=pm,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

# ---------------------------
# Submission writer & validator
# ---------------------------

def write_submission_csv(
    out_csv: pathlib.Path,
    ids: Sequence[str],
    mu: torch.Tensor,
    sigma: torch.Tensor,
    fmt: str = "wide",
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mu = mu.detach().cpu()
    sigma = sigma.detach().cpu()
    N, B = mu.shape
    if fmt not in {"wide", "long"}:
        raise ValueError("submission.format must be 'wide' or 'long'")

    if fmt == "wide":
        fieldnames = ["id"] + [f"mu_{i:03d}" for i in range(B)] + [
            f"sigma_{i:03d}" for i in range(B)
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row_idx in range(N):
                row = {"id": ids[row_idx]}
                for i in range(B):
                    row[f"mu_{i:03d}"] = float(mu[row_idx, i].item())
                for i in range(B):
                    val = float(max(0.0, sigma[row_idx, i].item()))
                    row[f"sigma_{i:03d}"] = val
                w.writerow(row)
    else:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "bin", "mu", "sigma"])
            w.writeheader()
            for row_idx in range(N):
                for i in range(B):
                    w.writerow(
                        {
                            "id": ids[row_idx],
                            "bin": i,
                            "mu": float(mu[row_idx, i].item()),
                            "sigma": float(max(0.0, sigma[row_idx, i].item())),
                        }
                    )


def validate_submission_csv(
    csv_path: pathlib.Path,
    fmt: str,
    bins: int,
    expect_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if fmt == "wide":
        required = {"id"} | {f"mu_{i:03d}" for i in range(bins)} | {
            f"sigma_{i:03d}" for i in range(bins)
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Submission missing columns: {sorted(missing)[:8]} ..."
            )
        ids = [r["id"] for r in rows]
        unique_ids = len(set(ids))
        if expect_ids is not None and set(ids) != set(expect_ids):
            raise ValueError("Submission IDs do not match expected test IDs.")
        return {
            "rows": len(rows),
            "unique_ids": unique_ids,
            "format": "wide",
            "bins": bins,
        }
    elif fmt == "long":
        if expect_ids is not None and len(rows) != len(expect_ids) * bins:
            raise ValueError("Long submission has incorrect row count.")
        for r in rows[:10]:
            for k in ["id", "bin", "mu", "sigma"]:
                if k not in r:
                    raise ValueError(f"Missing column {k} in long submission.")
        return {"rows": len(rows), "format": "long", "bins": bins}
    else:
        raise ValueError("Unknown format for validation.")


# ---------------------------
# Misc small helpers
# ---------------------------

def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, (list, tuple)):
            out[k] = [
                x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                for x in v
            ]
        elif isinstance(v, dict):
            out[k] = to_device(v, device)
        else:
            out[k] = v
    return out


def get_ids_from_batch(batch: Mapping[str, Any]) -> List[str]:
    for key in ("id", "ids", "planet_id", "planet_ids"):
        if key in batch:
            val = batch[key]
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val]
            if isinstance(val, torch.Tensor) and val.ndim == 1:
                return [str(x.item()) for x in val]
            return [str(val)]
    bs = None
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            bs = v.shape[0]
            break
        if isinstance(v, (list, tuple)):
            bs = len(v)
            break
    return [f"sample_{i:06d}" for i in range(bs or 0)]
