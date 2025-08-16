from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ConfigDict


def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 with 'Z' suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _detect_git_commit() -> str:
    """Best-effort detection of the current git commit hash.
    Uses environment variables commonly set in CI or falls back to '.git/HEAD' parsing if feasible.
    Returns 'unknown' if not found.
    """
    env_keys = [
        "GITHUB_SHA",
        "CI_COMMIT_SHA",
        "COMMIT_SHA",
        "GIT_COMMIT",
    ]
    for k in env_keys:
        v = os.environ.get(k)
        if v:
            return v
    # Lightweight attempt to parse .git plumbing if present
    head_path = os.path.join(os.getcwd(), ".git", "HEAD")
    try:
        if os.path.isfile(head_path):
            with open(head_path, "r", encoding="utf-8") as f:
                ref = f.read().strip()
            if ref.startswith("ref:"):
                ref_path = ref.split(" ", 1)[1].strip()
                ref_file = os.path.join(os.getcwd(), ".git", ref_path)
                if os.path.isfile(ref_file):
                    with open(ref_file, "r", encoding="utf-8") as f:
                        return f.read().strip()
            # Detached head may hold a raw commit in HEAD
            if len(ref) >= 7 and all(c in "0123456789abcdefABCDEF" for c in ref[:7]):
                return ref
    except Exception:
        pass
    return "unknown"


class BaseSchema(BaseModel):
    """Mission-grade base schema with reproducibility fields and robust (de)serialization helpers.

    Features:
    - Deterministic JSON serialization (sorted keys).
    - SHA256 content hash via stable canonical JSON.
    - Embedded metadata: schema_version, created_at (UTC), git_commit, run_hash (optional).
    - Strict field handling with Pydantic v2.
    """

    model_config = ConfigDict(
        extra="forbid",  # prevent silent typos
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=True,
        allow_inf_nan=False,  # enforce strict JSON
    )

    # --- Reproducibility meta ---
    schema_version: str = Field(default="1.0.0", description="Version of this schema object.")
    created_at: str = Field(default_factory=_utc_now_iso, description="UTC timestamp in ISO 8601 (Z).")
    git_commit: str = Field(default_factory=_detect_git_commit, description="Git commit hash if discoverable.")
    run_hash: Optional[str] = Field(
        default=None,
        description="Optional run hash or configuration hash tied to pipeline execution.",
    )

    # --- Core utilities ---
    def canonical_dict(self) -> Dict[str, Any]:
        """Return a dict ready for stable hashing/serialization (no Pydantic internals)."""
        # .model_dump is the canonical v2 way
        return self.model_dump(mode="json", by_alias=True, exclude_unset=False, exclude_none=False)

    def to_json(self, indent: int = 2) -> str:
        """Stable JSON with sorted keys; safe for hashing and diffs."""
        return json.dumps(self.canonical_dict(), sort_keys=True, ensure_ascii=False, indent=indent)

    def to_bytes(self) -> bytes:
        """Compact JSON bytes for hashing and storage."""
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def sha256(self) -> str:
        """SHA256 over the canonical JSON bytes."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_json(cls, data: str) -> "BaseSchema":
        """Load from JSON string."""
        obj = json.loads(data)
        return cls.model_validate(obj)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseSchema":
        """Load from Python dict."""
        return cls.model_validate(data)
