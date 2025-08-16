from __future__ import annotations

import datetime
import html
from pathlib import Path
from typing import Optional

from .config_audit_renderer import render_config_audit_section

HTML_HEAD = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>SpectraMind V50 Diagnostics Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { background:#0a0f1e; color:#e6f0ff; font: 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif; margin:0; }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 16px; }
  .card { background:#0e1530; border:1px solid #1f2a44; border-radius:12px; padding:16px; margin:16px 0; }
  h1 { font:800 24px/1.1 system-ui; margin: 0 0 8px; }
  p.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; color:#b7c7ff }
  a { color:#7aa2ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
<body>
<div class="wrap">
"""

HTML_FOOT = """
</div>
</body>
</html>
"""


def _card(title: str, body_html: str) -> str:
    return f"<div class='card'><h2 style='margin:0 0 10px'>{html.escape(title)}</h2>{body_html}</div>"


def generate_dashboard_html(
    out_html: str,
    *,
    title: str = "SpectraMind V50 Diagnostics Dashboard",
    config_audit_json: Optional[str] = None,
) -> str:
    """
    Generate an HTML diagnostics dashboard. For now, this focuses on embedding the
    Configuration Audit section, and is extensible to include other diagnostics.
    """

    now = datetime.datetime.utcnow().isoformat() + "Z"

    header = f"""
    <header class="card">
      <h1>{html.escape(title)}</h1>
      <p class="mono">Generated: {html.escape(now)}</p>
    </header>
    """

    # Config Audit Section
    config_section = ""
    if config_audit_json:
        config_section = _card(
            "Configuration", render_config_audit_section(config_audit_json)
        )
    else:
        config_section = _card("Configuration", "<p>No config audit JSON provided.</p>")

    html_out = HTML_HEAD + header + config_section + HTML_FOOT

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(html_out, encoding="utf-8")
    return out_html
