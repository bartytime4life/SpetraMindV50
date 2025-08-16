"""
SpectraMind V50 – Reporting Template Registry

This module provides a mission-grade, reproducible, and dependency-light interface
for discovering, rendering, and exporting Jinja2-based report templates that
ship inside the `spectramind.reporting.report_templates` package.

Design goals:

* Zero placeholder philosophy: all shipped templates render with a minimal context.
* Reproducibility: deterministic defaults, explicit environment configuration,
  and content-hash helpers for artifact manifests.
* Safety: HTML autoescape when rendering HTML/JSON templates; conservative defaults for Markdown.
* Usability: single entrypoint `get_registry()`; convenient module-level helpers.

Templates and assets live in this package:

* *.html.j2, *.md.j2, *.json.j2 – Jinja2 templates
* assets/ – CSS/JS used by HTML reports

Typical usage:
from spectramind.reporting.report_templates import get_registry
reg = get_registry()
html = reg.render_to_string("diagnostic_report.html.j2", context)
reg.render_to_file("diagnostic_report.html.j2", context, "out/report.html", copy_assets=True)

CLI/system integration:

* This module is importable anywhere; no side effects at import.
* Uses importlib.resources for safe access when installed as a wheel/zip.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources as ires
from typing import Any, Dict, List, Mapping, Optional

try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound
except Exception as e:  # pragma: no cover - import error is actionable
    raise RuntimeError(
        "Jinja2 is required for SpectraMind reporting templates.\n"
        "Install with: pip install jinja2"
    ) from e

_LOG = logging.getLogger("spectramind.reporting.template_registry")


def _package_root_path() -> str:
    """Resolve a real filesystem path to this package's root directory, even if
    installed from a wheel/zip. Falls back to a temporary extracted dir."""
    pkg = "spectramind.reporting.report_templates"
    with ires.as_file(ires.files(pkg)) as p:
        return str(p)


def _autoescape_for_template(name: Optional[str]) -> bool:
    """Enable autoescape for HTML/XML/JSON templates. Our templates typically end with:
    - .html.j2, .xml.j2, .json.j2  → autoescape = True
    - .md.j2                        → autoescape = False"""
    if not name:
        return False
    name = name.lower()
    return name.endswith((".html.j2", ".xml.j2", ".json.j2"))


def _default_filters() -> Dict[str, Any]:
    """Register conservative, deterministic filters for templates."""

    def fmt_dt(dt: Any, fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
        if isinstance(dt, (int, float)):
            dt = datetime.fromtimestamp(float(dt), tz=timezone.utc)
        elif isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except Exception:
                return dt
        if isinstance(dt, datetime) and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime(fmt)

    def sha256_text(s: Any) -> str:
        return hashlib.sha256(str(s).encode("utf-8")).hexdigest()

    def pretty_json(obj: Any, indent: int = 2, sort_keys: bool = True) -> str:
        return json.dumps(obj, indent=indent, sort_keys=sort_keys)

    def clamp(v: Any, lo: float = 0.0, hi: float = 1.0) -> Any:
        try:
            x = float(v)
        except Exception:
            return v
        return max(lo, min(hi, x))

    def md_escape(s: str) -> str:
        return re.sub(r"([\\`*_{}\[\]()#+\-.!])", r"\\\1", s)

    return {
        "fmt_dt": fmt_dt,
        "sha256": sha256_text,
        "pretty_json": pretty_json,
        "clamp": clamp,
        "md_escape": md_escape,
    }


@dataclass(frozen=True)
class TemplateInfo:
    """Metadata for a packaged report template."""

    name: str
    kind: str  # "html" | "md" | "json" | "other"
    path: str  # absolute filesystem path
    description: str


class TemplateRegistry:
    """
    Registry and renderer for packaged SpectraMind report templates.

    Responsibilities:
    - Discover all *.j2 templates in this package.
    - Render templates to strings/files with safe defaults.
    - Copy static assets (CSS/JS) alongside rendered HTML.
    """

    def __init__(self) -> None:
        self._root = _package_root_path()
        self._loader = FileSystemLoader(self._root)
        self._env = Environment(
            loader=self._loader,
            autoescape=lambda name: _autoescape_for_template(name),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        for k, v in _default_filters().items():
            self._env.filters[k] = v
        # Global functions available in templates:
        self._env.globals.update(
            {
                "now_utc": lambda: datetime.now(tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                ),
                "inline_asset": self.inline_asset,
            }
        )

    # -------------------------
    # Discovery
    # -------------------------

    def list_templates(self) -> List[TemplateInfo]:
        """
        Return metadata for all templates in this package.
        Kinds are inferred from filename suffix: *.html.j2 → "html", etc.
        """
        templates: List[TemplateInfo] = []
        for dirpath, _, filenames in os.walk(self._root):
            for fn in filenames:
                if not fn.endswith(".j2"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, self._root).replace("\\", "/")
                kind = "other"
                if fn.endswith(".html.j2"):
                    kind = "html"
                elif fn.endswith(".md.j2"):
                    kind = "md"
                elif fn.endswith(".json.j2"):
                    kind = "json"
                desc = self._infer_description(rel)
                templates.append(
                    TemplateInfo(name=rel, kind=kind, path=full, description=desc)
                )
        templates.sort(key=lambda t: (t.kind, t.name))
        return templates

    def _infer_description(self, rel_path: str) -> str:
        """Best-effort description based on filename."""
        base = rel_path.lower()
        if "diagnostic_report.html.j2" in base:
            return "Unified diagnostics dashboard (HTML) for UMAP/t-SNE, GLL, symbolic overlays, and runtime metadata."
        if "diagnostic_summary.md.j2" in base:
            return "Markdown summary of diagnostics metrics, calibration, and key artifacts."
        if "cli_log_summary.md.j2" in base:
            return "Markdown table view of recent CLI calls from v50_debug_log.md."
        if "symbolic_rule_table.html.j2" in base:
            return "HTML table of top symbolic rule violations per planet."
        if "umap_embed.html.j2" in base:
            return "HTML partial to embed an interactive UMAP plot."
        if "tsne_embed.html.j2" in base:
            return "HTML partial to embed an interactive t-SNE plot."
        if "gll_heatmap.html.j2" in base:
            return (
                "HTML partial to show a bin-wise GLL heatmap (supports base64 image)."
            )
        if "manifest.json.j2" in base:
            return "JSON manifest for report artifacts (reproducibility and CI integration)."
        if "report_bundle.json.j2" in base:
            return "JSON bundle descriptor aggregating report pages and assets."
        if "base.html.j2" in base:
            return "Base HTML layout with SpectraMind styles and macros."
        if "macros.html.j2" in base:
            return "Reusable macros (tables, badges, chips, progress bars)."
        return f"Template: {rel_path}"

    # -------------------------
    # Rendering
    # -------------------------

    def get_environment(self) -> Environment:
        """Return the underlying Jinja2 environment."""
        return self._env

    def render_to_string(
        self, template_name: str, context: Optional[Mapping[str, Any]] = None
    ) -> str:
        """
        Render the given template to a string.

        Args:
            template_name: Relative path under this package, e.g., 'diagnostic_report.html.j2'
            context: Variables for the template.

        Returns:
            Rendered string.
        """
        context = dict(context or {})
        context.setdefault(
            "generated_at_utc", datetime.now(tz=timezone.utc).isoformat()
        )
        context.setdefault("spectramind_version", "V50")
        try:
            tmpl = self._env.get_template(template_name)
        except TemplateNotFound as e:
            raise FileNotFoundError(f"Template not found: {template_name}") from e
        return tmpl.render(**context)

    def render_to_file(
        self,
        template_name: str,
        context: Optional[Mapping[str, Any]],
        output_path: str,
        *,
        copy_assets: bool = False,
        ensure_parent: bool = True,
        mode: int = 0o644,
        encoding: str = "utf-8",
    ) -> str:
        """
        Render a template and write to `output_path`.

        Args:
            template_name: template file name (relative to package root)
            context: variables to render with
            output_path: destination file path
            copy_assets: if True, copy assets/ into output dir and adjust relative paths if needed
            ensure_parent: create parent directories if missing
            mode: chmod bits for the written file
            encoding: file encoding

        Returns:
            Absolute path to the written file.
        """
        s = self.render_to_string(template_name, context)
        out = os.path.abspath(output_path)
        if ensure_parent:
            os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding=encoding) as f:
            f.write(s)
        os.chmod(out, mode)
        if copy_assets:
            self.copy_assets(os.path.dirname(out))
        _LOG.info("Rendered %s → %s", template_name, out)
        return out

    # -------------------------
    # Assets
    # -------------------------

    def assets_root(self) -> str:
        """Absolute path to the packaged assets directory."""
        return os.path.join(self._root, "assets")

    def copy_assets(self, dest_dir: str) -> str:
        """
        Copy packaged assets to `<dest_dir>/assets`, replacing existing files if any.

        Returns:
            Absolute path to the destination assets directory.
        """
        src = self.assets_root()
        dst = os.path.join(os.path.abspath(dest_dir), "assets")
        os.makedirs(dst, exist_ok=True)
        for dirpath, _, filenames in os.walk(src):
            rel = os.path.relpath(dirpath, src)
            out_dir = os.path.join(dst, rel) if rel != "." else dst
            os.makedirs(out_dir, exist_ok=True)
            for fn in filenames:
                shutil.copy2(os.path.join(dirpath, fn), os.path.join(out_dir, fn))
        _LOG.info("Copied assets → %s", dst)
        return dst

    def inline_asset(self, rel_path: str) -> str:
        """
        Return the text content of an asset in the package, for inline embedding
        (e.g., CSS/JS in an HTML template).

        Example:
            {{ inline_asset('assets/spectramind_report.css') }}
        """
        p = os.path.join(self._root, rel_path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Asset not found: {rel_path}")
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    # -------------------------
    # Utilities
    # -------------------------

    def content_hash(self, data: str) -> str:
        """SHA256 hash for content provenance and manifest linking."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def minimal_context(self) -> Dict[str, Any]:
        """
        Provide a deterministic, valid context that renders every shipped template,
        useful for smoke-tests and self-tests.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        return {
            "generated_at_utc": now,
            "spectramind_version": "V50",
            "run": {
                "config_hash": "0000000000000000000000000000000000000000000000000000000000000000",
                "git_commit": "unknown",
                "cli": "spectramind --version V50",
                "host": "local",
                "env": {"python": "3.x", "cuda": "n/a"},
            },
            "metrics": {
                "gll_mean": 0.0,
                "rmse_mean": 0.0,
                "calibration_error": 0.0,
                "num_planets": 0,
                "duration_seconds": 0.0,
            },
            "artifacts": {
                "umap_html": "<div id='umap'>UMAP preview</div>",
                "tsne_html": "<div id='tsne'>t-SNE preview</div>",
                "gll_heatmap_base64": "",
                "symbolic_rule_table_rows": [],
            },
            "cli_log_rows": [],
        }


# -------------------------
# Module-level convenience
# -------------------------

_singleton: Optional[TemplateRegistry] = None


def get_registry() -> TemplateRegistry:
    """Return a module-level singleton TemplateRegistry."""
    global _singleton
    if _singleton is None:
        _singleton = TemplateRegistry()
    return _singleton


def list_templates() -> List[TemplateInfo]:
    """List all templates available in this package."""
    return get_registry().list_templates()


def render_template_to_string(
    template_name: str, context: Optional[Mapping[str, Any]] = None
) -> str:
    """Convenience wrapper for TemplateRegistry.render_to_string()."""
    return get_registry().render_to_string(template_name, context)


def render_template_to_file(
    template_name: str,
    context: Optional[Mapping[str, Any]],
    output_path: str,
    *,
    copy_assets: bool = False,
    ensure_parent: bool = True,
) -> str:
    """Convenience wrapper for TemplateRegistry.render_to_file()."""
    return get_registry().render_to_file(
        template_name,
        context,
        output_path,
        copy_assets=copy_assets,
        ensure_parent=ensure_parent,
    )


def render_inline_asset(rel_path: str) -> str:
    """Read a packaged asset as text; useful for tests and external embedding."""
    return get_registry().inline_asset(rel_path)


def copy_assets(dest_dir: str) -> str:
    """Copy packaged assets/ into dest_dir/assets."""
    return get_registry().copy_assets(dest_dir)
