# SPDX-License-Identifier: MIT

"""Configuration validation utilities."""

from typing import Any, Dict, Union
from omegaconf import OmegaConf, DictConfig

from .schema import V50ConfigSchema
from .logging_utils import write_md, log_event
from .hashing import config_hash
from .exceptions import ConfigValidationError


def validate_config(config: Union[Dict[str, Any], DictConfig]) -> V50ConfigSchema:
    """Validate a configuration against the Pydantic schema with logging."""
    try:
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        schema_obj = V50ConfigSchema(**config)  # type: ignore[arg-type]
        h = config_hash(config)
        write_md("Config validation successful", {"hash": h})
        log_event("config_validate_ok", {"hash": h})
        return schema_obj
    except Exception as e:  # pragma: no cover - pydantic validation
        write_md("Config validation failed", {"error": str(e)})
        log_event("config_validate_error", {"error": str(e)})
        raise ConfigValidationError(str(e)) from e
