from __future__ import annotations

import html
import json
from pathlib import Path


def _badge(text: str, ok: bool) -> str:
    color = "#16a34a" if ok else "#dc2626"
    return f'<span style="background:{color};color:#fff;border-radius:8px;padding:4px 8px;font:600 12px/1.2 system-ui">{html.escape(text)}</span>'


def _kv(k: str, v: str) -> str:
    return f"<tr><td style='font-weight:600;padding:6px 8px;width:220px;vertical-align:top'>{html.escape(k)}</td><td style='padding:6px 8px'>{v}</td></tr>"


def _pre(obj: object) -> str:
    j = json.dumps(obj, indent=2)
    return f"<pre style='background:#0b1020;color:#e6f0ff;border:1px solid #1f2a44;border-radius:8px;padding:12px;overflow:auto;max-height:420px'>{html.escape(j)}</pre>"


def render_config_audit_section(audit_json_path: str) -> str:
    """
    Render an HTML section for the config audit JSON.
    """
    p = Path(audit_json_path)
    if not p.exists():
        return "<!-- config audit JSON not found; skipping section -->"

    data = json.loads(p.read_text())

    ok = bool(data.get("ok", False))
    config_source = str(data.get("config_source", ""))
    schema_source = (
        str(data.get("schema_source", "")) if data.get("schema_source") else "None"
    )
    validation_error = data.get("validation_error") or "None"
    has_symbolic = bool(data.get("has_symbolic_defaults", False))

    head = f"""
    <section id="config-audit" style="margin:18px 0 28px 0">
      <h2 style="font:700 20px/1.2 system-ui;margin:0 0 8px">Configuration Audit</h2>
      <div style="margin:6px 0 12px">{_badge("VALID" if ok else "INVALID", ok)}</div>
      <table style="border-collapse:collapse;width:100%;border:1px solid #1f2a44;border-radius:8px;overflow:hidden">
        {_kv("Config Source", html.escape(config_source))}
        {_kv("Schema Source", html.escape(schema_source))}
        {_kv("Symbolic Defaults Injected", "Yes" if has_symbolic else "No")}
        {_kv("Validation Error", html.escape(validation_error))}
      </table>
    </section>
    """

    resolved = data.get("resolved_sample", {})
    env = data.get("environment", {})

    body = f"""
    <section style="margin:12px 0 20px 0">
      <h3 style="font:700 16px/1.2 system-ui;margin:0 0 8px">Resolved Config (sample)</h3>
      {_pre(resolved)}
    </section>
    <section style="margin:12px 0 20px 0">
      <h3 style="font:700 16px/1.2 system-ui;margin:0 0 8px">Environment Capture</h3>
      {_pre(env)}
    </section>
    """

    return head + body
