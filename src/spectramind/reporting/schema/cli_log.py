from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field

from .base import BaseSchema
from .types import HashStr, IsoDatetime, NonEmptyStr, PathLikeStr
from .schema_registry import register_model


@register_model
class CLILogEntry(BaseSchema):
    """One CLI call record ingested from v50_debug_log.md or internal event stream."""

    timestamp: IsoDatetime = Field(description="When the CLI call started.")
    cli: NonEmptyStr = Field(description="Root CLI name (e.g., 'spectramind').")
    subcommand: NonEmptyStr = Field(description="Subcommand invoked (e.g., 'diagnose dashboard').")
    args: List[str] = Field(default_factory=list, description="CLI argv tail (sanitized).")
    config_hash: Optional[HashStr] = Field(default=None, description="Config hash associated with run.")
    run_id: Optional[HashStr] = Field(default=None, description="Run identifier (if present).")
    exit_code: int = Field(default=0, ge=0, description="Process exit code (0 = success).")
    duration_s: float = Field(default=0.0, ge=0.0, description="Wall-clock duration in seconds.")
    host: Optional[NonEmptyStr] = Field(default=None, description="Hostname or node ID.")
    user: Optional[NonEmptyStr] = Field(default=None, description="OS user.")
    cwd: Optional[PathLikeStr] = Field(default=None, description="Working directory.")
    tags: List[NonEmptyStr] = Field(default_factory=list, description="Freeform labels (e.g., 'ci', 'local').")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Key metrics (e.g., gll=..., rmse=...).")


@register_model
class CLILogBatch(BaseSchema):
    """A batch/slice of CLI log entries plus local aggregates suitable for dashboard summaries."""

    entries: List[CLILogEntry] = Field(default_factory=list, description="Ordered log entries.")
    by_subcommand_count: Dict[str, int] = Field(default_factory=dict, description="Subcommand → count.")
    by_config_hash_count: Dict[str, int] = Field(default_factory=dict, description="ConfigHash → count.")
    total_calls: int = Field(default=0, ge=0, description="Total number of calls summarized.")
