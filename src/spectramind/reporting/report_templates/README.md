# SpectraMind V50 – Report Templates

This package contains the mission-ready Jinja2 templates and assets used to generate HTML/Markdown/JSON reports across the SpectraMind V50 pipeline (diagnostics dashboard, CLI log summaries, symbolic rule tables, manifests).

Contents:

* `base.html.j2` – Base HTML document with built-in dark-aware styling and layout primitives.
* `macros.html.j2` – Table/badge macros for consistent visual components.
* `diagnostic_report.html.j2` – Full HTML diagnostics dashboard integrating UMAP/t-SNE, GLL heatmaps, symbolic rule tables, and run metadata.
* `diagnostic_summary.md.j2` – Markdown summary of core metrics and artifacts.
* `cli_log_summary.md.j2` – Markdown table summarizing recent CLI calls (for `spectramind analyze-log`).
* `symbolic_rule_table.html.j2` – Sortable table of top symbolic rule violations per planet.
* `umap_embed.html.j2`, `tsne_embed.html.j2`, `gll_heatmap.html.j2` – Embeddable partials.
* `manifest.json.j2`, `report_bundle.json.j2` – Reproducibility manifests for CI/export.
* `assets/` – CSS/JS used by HTML templates.

Usage (Python):
```python
from spectramind.reporting.report_templates import get_registry
reg = get_registry()
ctx = reg.minimal_context()  # or your real context
reg.render_to_file("diagnostic_report.html.j2", ctx, "out/report.html", copy_assets=True)
```

Hydra example (config snippet):
```
reporting:
  template: diagnostic_report.html.j2
  output: ${paths.out_dir}/diagnostics/report.html
  copy_assets: true
```

Packaging note:
Ensure your packaging includes these files as package data (e.g., with setuptools `include_package_data=True` and `MANIFEST.in` rules), so that `importlib.resources` can access them at runtime.
