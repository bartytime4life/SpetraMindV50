# SPDX-License-Identifier: MIT

"""SpectraMind V50 configuration helper package."""

from .exceptions import (
    ConfigValidationError,
    OverrideLoadError,
    HydraLoadError,
    SchemaExportError,
    ConfHelpersError,
)

from .schema import V50ConfigSchema, get_json_schema
from .hashing import config_hash, stable_dumps
from .io import load_yaml, save_yaml, load_json, save_json, atomic_write
from .logging_utils import (
    get_logger,
    log_event,
    init_logging,
    LOG_JSONL,
    V50_DEBUG_LOG,
)
from .hydra_integration import load_config_hydra
from .validators import validate_config
from .overrides import apply_symbolic_overrides, deep_update, load_overrides_layered
from .loader import load_and_validate

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
    "validate_config",
    "apply_symbolic_overrides",
    "deep_update",
    "load_overrides_layered",
    "load_and_validate",
]
