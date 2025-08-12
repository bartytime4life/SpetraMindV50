from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def write_report(path: Path) -> None:
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>SpectraMind V50 — Diagnostics</title>
  <style>body{{font-family:system-ui,Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}}.card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:12px}}</style>
</head>
<body>
  <h1>Diagnostics — SpectraMind V50</h1>
  <div class=\"card\">
    <h2>Run Metadata</h2>
    <p>Generated: {ts}</p>
  </div>
  <div class=\"card\">
    <h2>Placeholders</h2>
    <ul>
      <li>UMAP/t-SNE: pending</li>
      <li>SHAP × symbolic overlays: pending</li>
      <li>FFT/smoothness maps: pending</li>
    </ul>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
