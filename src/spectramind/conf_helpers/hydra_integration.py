# SPDX-License-Identifier: MIT

"""Hydra composition helpers."""

from pathlib import Path
from typing import Optional, List
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir

from .logging_utils import get_logger, write_md, log_event
from .exceptions import HydraLoadError


def load_config_hydra(
    config_dir: str | Path,
    config_name: str = "config_v50.yaml",
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load a configuration using Hydra with logging and error handling."""
    logger = get_logger()
    config_dir = Path(config_dir).resolve()
    try:
        meta = {
            "config_dir": str(config_dir),
            "config_name": config_name,
            "overrides": overrides or [],
        }
        write_md("Loading Hydra config", meta)
        log_event("hydra_load_begin", meta)
        with initialize_config_dir(config_dir=str(config_dir), job_name="spectramind_conf_loader"):
            cfg = compose(config_name=config_name, overrides=overrides or [])
        log_event("hydra_load_ok", {"config_name": config_name})
        return cfg
    except Exception as e:  # pragma: no cover - hydra provides rich errors
        logger.exception("Hydra load failed: %s", e)
        log_event("hydra_load_error", {"error": str(e)})
        write_md("Hydra load failed", {"error": str(e)})
        raise HydraLoadError(str(e)) from e
