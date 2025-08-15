import os
import io
import logging
import typing as T
from dataclasses import dataclass

# Optional HTML-to-PDF backends
try:
    import weasyprint  # type: ignore
    _HAS_WEASY = True
except Exception:
    _HAS_WEASY = False

try:
    import pdfkit  # requires wkhtmltopdf installed
    _HAS_PDFKIT = True
except Exception:
    _HAS_PDFKIT = False


@dataclass
class ExportConfig:
    """Controls output behaviors for report exporting."""
    ensure_self_contained_html: bool = True
    # Reserved for future options (asset inlining, CDN policy, etc.)


class ExportManager:
    """Handles saving HTML, Markdown, and (best-effort) PDF exports."""

    def __init__(self, cfg: ExportConfig = ExportConfig(), logger: logging.Logger = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def save_html(self, path: str, html: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        self.logger.info(f"Saved HTML: {path}")

    def save_markdown(self, path: str, md: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        self.logger.info(f"Saved Markdown: {path}")

    def save_pdf(self, path: str, html: str) -> None:
        """Best-effort HTML → PDF export. Prefer WeasyPrint; fallback to pdfkit; else skip with warning."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if _HAS_WEASY:
            try:
                weasyprint.HTML(string=html).write_pdf(path)
                self.logger.info(f"Saved PDF (WeasyPrint): {path}")
                return
            except Exception as e:
                self.logger.warning(f"WeasyPrint PDF export failed: {e}")
        if _HAS_PDFKIT:
            try:
                pdfkit.from_string(html, path)
                self.logger.info(f"Saved PDF (pdfkit): {path}")
                return
            except Exception as e:
                self.logger.warning(f"pdfkit PDF export failed: {e}")
        self.logger.warning("No PDF engine available; skipping PDF export.")
