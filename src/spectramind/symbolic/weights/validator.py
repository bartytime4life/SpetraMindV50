"""Schema validation for weights and profiles.

We keep dependencies minimal (no pydantic) to remain challenge-safe.
Rules:
    • All top-level values numeric >= 0 OR nested dicts with numeric leaves >= 0
    • Known keys are not strictly enforced to allow forward-compat config, but we log unknown types
    • Profiles may be nested; validation recurses
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .logging_utils import emit_event, get_logger

LOGGER = get_logger(__name__)


@dataclass
class WeightSchema:
    allow_nested: bool = True
    require_non_negative: bool = True


@dataclass
class WeightProfileSchema:
    """Schema for entries under composite_profile_weights or standalone profiles/*."""

    allow_nested: bool = True
    require_non_negative: bool = True


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_map(
    m: Dict[str, Any],
    allow_nested: bool,
    require_non_negative: bool,
    path: Tuple[str, ...] = (),
    errors: List[str] | None = None,
) -> List[str]:
    errs: List[str] = [] if errors is None else errors
    for k, v in m.items():
        cur = path + (k,)
        if isinstance(v, dict):
            if not allow_nested:
                errs.append(f"Nested mapping not allowed at {'.'.join(cur)}")
            else:
                _validate_map(v, allow_nested, require_non_negative, cur, errs)
        else:
            if not _is_number(v):
                errs.append(
                    f"Non-numeric weight at {'.'.join(cur)}: {type(v).__name__}"
                )
            elif require_non_negative and v < 0:
                errs.append(f"Negative weight at {'.'.join(cur)}: {v}")
    return errs


def validate_weight_config(
    cfg: Dict[str, Any], schema: WeightSchema | None = None
) -> None:
    sch = schema or WeightSchema()
    errs = _validate_map(cfg, sch.allow_nested, sch.require_non_negative)
    if errs:
        for e in errs:
            LOGGER.error("weight_validation_error", extra={"error": e})
        emit_event("weights_validation_failed", {"errors": errs})
        raise ValueError("Invalid weight config:\n" + "\n".join(errs))
    LOGGER.info("weight_validation_ok", extra={"num_keys": len(cfg)})
    emit_event("weights_validation_ok", {"num_keys": len(cfg)})


def validate_profile_weights(
    cfg: Dict[str, Any], schema: WeightProfileSchema | None = None
) -> None:
    sch = schema or WeightProfileSchema()
    validate_weight_config(
        cfg,
        WeightSchema(
            allow_nested=sch.allow_nested, require_non_negative=sch.require_non_negative
        ),
    )
