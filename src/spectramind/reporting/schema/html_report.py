from __future__ import annotations

import os
from typing import List, Optional

from pydantic import Field

from .base import BaseSchema
from .types import NonEmptyStr, PathLikeStr
from .schema_registry import register_model


@register_model
class HTMLAsset(BaseSchema):
    """One HTML-embeddable asset (plot, iframe, table) used by the diagnostics dashboard."""

    kind: NonEmptyStr = Field(description="Type (e.g., 'plotly', 'png', 'iframe', 'table', 'csv').")
    title: NonEmptyStr = Field(description="Human-readable title.")
    path: PathLikeStr = Field(description="Relative or absolute path to the asset.")
    mime: Optional[str] = Field(default=None, description="Optional MIME type (e.g., 'text/html', 'image/png').")
    width: Optional[int] = Field(default=None, ge=1, description="Preferred pixel width for layout.")
    height: Optional[int] = Field(default=None, ge=1, description="Preferred pixel height for layout.")


@register_model
class HTMLReportManifest(BaseSchema):
    """Top-level HTML report manifest to drive the renderer and integrity checks."""

    title: NonEmptyStr = Field(description="Report title.")
    assets: List[HTMLAsset] = Field(default_factory=list, description="List of assets, in display order.")
    description: Optional[str] = Field(default=None, description="Optional description / synopsis.")

    def verify_files(self, base_dir: Optional[str] = None) -> List[str]:
        """Return list of missing asset paths (resolved against base_dir if provided)."""
        missing: List[str] = []
        for a in self.assets:
            p = a.path
            if base_dir is not None and not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            if not os.path.exists(p):
                missing.append(a.path)
        return missing
