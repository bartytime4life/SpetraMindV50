# SPDX-License-Identifier: MIT

"""SpectraMind V50 configuration helper package."""

from .config_loader import load_config, save_config
from .env_capture import capture_environment, log_environment
from .exceptions import (
    ConfHelpersError,
    ConfigValidationError,
    HydraLoadError,
    OverrideLoadError,
    SchemaExportError,
)
from .hashing import config_hash, stable_dumps
from .hydra_integration import load_config_hydra
from .io import atomic_write, load_json, load_yaml, save_json, save_yaml
from .loader import load_and_validate
from .logging_utils import LOG_JSONL, V50_DEBUG_LOG, get_logger, init_logging, log_event
from .overrides import (
    apply_overrides,
    apply_symbolic_overrides,
    cli_override_parser,
    deep_update,
    load_overrides_layered,
)
from .schema import V50ConfigSchema, get_json_schema
from .schema_validator import validate_config
from .symbolic_hooks import inject_symbolic_constraints
from .validators import validate_config as validate_config_pydantic

__all__ = [
    # Exceptions
    "ConfHelpersError",
    "ConfigValidationError",
    "OverrideLoadError",
    "HydraLoadError",
    "SchemaExportError",
    # Core
    "V50ConfigSchema",
    "get_json_schema",
    "config_hash",
    "stable_dumps",
    "load_yaml",
    "save_yaml",
    "load_json",
    "save_json",
    "atomic_write",
    "get_logger",
    "log_event",
    "init_logging",
    "LOG_JSONL",
    "V50_DEBUG_LOG",
    "load_config_hydra",
    "load_config",
    "save_config",
    "validate_config",
    "validate_config_pydantic",
    "apply_symbolic_overrides",
    "deep_update",
    "cli_override_parser",
    "apply_overrides",
    "load_overrides_layered",
    "inject_symbolic_constraints",
    "capture_environment",
    "log_environment",
    "load_and_validate",
]
