from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from .config_loader import load_config
from .env_capture import capture_environment_detailed
from .schema_validator import validate_config
from .symbolic_hooks import inject_symbolic_constraints


@dataclass
class ConfigAuditResult:
    ok: bool
    config_source: str
    schema_source: Optional[str]
    validation_error: Optional[str]
    has_symbolic_defaults: bool
    resolved_sample: Dict[str, Any]
    environment: Dict[str, Any]


def run_config_audit(
    config_path_or_name: str,
    schema_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    out_json: Optional[str] = None,
) -> ConfigAuditResult:
    """
    Load config, optionally apply overrides, validate against schema, ensure symbolic defaults,
    and capture environment for the dashboard.

    Writes JSON if out_json is provided.
    """
    overrides = overrides or []
    validation_error = None
    ok = True

    cfg = load_config(config_path_or_name, overrides)
    cfg = inject_symbolic_constraints(cfg)

    if schema_path:
        try:
            validate_config(cfg, schema_path)
        except Exception as e:
            ok = False
            validation_error = f"{type(e).__name__}: {e}"

    resolved = OmegaConf.to_container(cfg, resolve=True)
    env = capture_environment_detailed()

    result = ConfigAuditResult(
        ok=ok,
        config_source=config_path_or_name,
        schema_source=schema_path,
        validation_error=validation_error,
        has_symbolic_defaults=True,
        resolved_sample=(
            resolved if isinstance(resolved, dict) else {"config": resolved}
        ),
        environment=env,
    )

    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(asdict(result), f, indent=2)

    return result
