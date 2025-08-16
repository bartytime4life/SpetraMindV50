"""
SpectraMind V50 - conf_helpers package

Mission-grade utilities for robust Hydra/YAML configuration handling, schema validation,
symbolic/physics defaults injection, and environment capture for reproducibility.
"""

from .config_audit import ConfigAuditResult, run_config_audit
from .config_loader import (
    get_hydra_compose,
    load_config,
    load_yaml,
    resolve_config_source,
    save_config,
)
from .env_capture import (
    capture_environment,
    capture_environment_detailed,
    log_environment,
)
from .overrides import apply_overrides, cli_override_parser
from .schema_validator import validate_config
from .symbolic_hooks import inject_symbolic_constraints

__all__ = [
    "load_config",
    "load_yaml",
    "save_config",
    "get_hydra_compose",
    "resolve_config_source",
    "apply_overrides",
    "cli_override_parser",
    "validate_config",
    "inject_symbolic_constraints",
    "capture_environment",
    "log_environment",
    "capture_environment_detailed",
    "ConfigAuditResult",
    "run_config_audit",
]
