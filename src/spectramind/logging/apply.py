# src/spectramind/logging/apply.py
# Mission-grade logging installer for Hydra-driven configs.
# - Reads OmegaConf dict under cfg.logging (Hydra) and applies logging configuration.
# - Auto-creates logs/ directory and appends a one-line run header to v50_debug_log.md when file-based logging is active.
# - Safe to call multiple times (idempotent-ish): it replaces logging config each call via dictConfig.
# - Writes a compact version line including timestamp, CLI argv, and working dir.

from __future__ import annotations
import os
import sys
import json
import time
import socket
import getpass
import pathlib
import logging
import logging.config
from typing import Any, Mapping

def _ensure_parent(path: str) -> None:
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

def _append_run_header_if_file_logging(logging_cfg: Mapping[str, Any]) -> None:
    try:
        handlers = logging_cfg.get("handlers", {}) if isinstance(logging_cfg, dict) else {}
        for hname, hcfg in handlers.items():
            if not isinstance(hcfg, dict):
                continue
            cls = hcfg.get("class", "")
            filename = hcfg.get("filename")
            if filename and ("FileHandler" in cls or "RotatingFileHandler" in cls):
                fpath = pathlib.Path(filename).expanduser()
                _ensure_parent(str(fpath))
                # Also ensure logs dir exists for our standard md file if used
                if fpath.parent.name == "logs":
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                # Append concise header
                header = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "user": getpass.getuser(),
                    "host": socket.gethostname(),
                    "cwd": os.getcwd(),
                    "argv": sys.argv,
                }
                try:
                    with open(fpath, "a", encoding="utf-8") as f:
                        f.write(f"\n\n--- RUN HEADER --- {json.dumps(header, ensure_ascii=False)}\n")
                except Exception:
                    # Do not fail pipeline for log header write
                    pass
    except Exception:
        pass

def install_from_hydra(cfg: Any, *, raise_on_missing: bool = True) -> None:
    """
    Install Python logging configuration from a Hydra config object.

    Parameters
    ----------
    cfg : Any
        Hydra/OmegaConf config object expected to contain a `logging` dictConfig spec.
    raise_on_missing : bool
        If True, raises when `cfg.logging` is absent or not a mapping. Otherwise, returns silently.

    Notes
    -----
    - This function is safe to call early in your entrypoints (e.g., train_v50.py, predict_v50.py).
    - When using file logging, the function ensures the target directory exists and appends a small
      JSON header to the log file to aid diagnostics and CLI usage analysis.
    """
    # OmegaConf -> dict compatibility
    log_cfg = getattr(cfg, "logging", None)
    if log_cfg is None:
        if raise_on_missing:
            raise RuntimeError("Hydra config missing `logging` section; cannot install logging.")
        return

    # If OmegaConf, convert to raw dict
    try:
        import omegaconf
        if isinstance(log_cfg, omegaconf.DictConfig):
            from omegaconf import OmegaConf
            log_cfg = OmegaConf.to_container(log_cfg, resolve=True)
    except Exception:
        pass

    if not isinstance(log_cfg, dict):
        if raise_on_missing:
            raise TypeError("`cfg.logging` is not a dict mapping suitable for dictConfig.")
        return

    # Ensure file handler parents exist + append header line
    _append_run_header_if_file_logging(log_cfg)

    # Install configuration
    logging.config.dictConfig(log_cfg)

    # Friendly root logger message (optional)
    logging.getLogger(__name__).debug("Installed logging via Hydra config.")
