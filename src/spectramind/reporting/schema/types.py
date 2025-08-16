from __future__ import annotations

import re
from typing import Annotated

from pydantic import Field

# --- String constraints ---

# A non-empty, trimmed string
NonEmptyStr = Annotated[str, Field(strict=True, min_length=1, strip_whitespace=True)]

# ISO 8601 datetime string with Z or offset. We do not fully parse here; we validate shape lightly.
_ISO_DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
IsoDatetime = Annotated[
    str,
    Field(
        pattern=_ISO_DT_RE.pattern,
        description="ISO 8601 datetime (e.g., 2025-08-16T03:04:05Z).",
    ),
]

# A "hash-like" string used for commits, config hashes, or run IDs.
# Accept hex (7-64 chars) or base64url-ish (>=8)
_HASH_RE = re.compile(r"^(?:[0-9a-fA-F]{7,64}|[A-Za-z0-9_-]{8,})$")
HashStr = Annotated[
    str,
    Field(pattern=_HASH_RE.pattern, description="Commit or config hash; hex(7..64) or base64url-ish."),
]

# Planet identifiers seen in challenge metadata, tolerant to A-Z,a-z,0-9,_, -, .
PlanetId = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9_.-]+$",
        min_length=1,
        max_length=128,
        description="Planet identifier (alnum, '-', '_', '.').",
    ),
]

# Path-like string (not validated to exist, as artifacts may be produced later).
PathLikeStr = Annotated[
    str,
    Field(min_length=1, description="Filesystem path or POSIX-like path as string."),
]
