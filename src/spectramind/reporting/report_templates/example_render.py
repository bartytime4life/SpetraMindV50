"""
Example renderer for SpectraMind report templates.

This script demonstrates how to render the mission-ready templates with either:

* A minimal, deterministic context (registry.minimal_context())
* A custom realistic context (populate below)

Run:
python -m spectramind.reporting.report_templates.example_render

Outputs are written to ./_example_out by default.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from .template_registry import get_registry

# Optional: create a small transparent PNG to prove base64 embedding works.

_PNG_DOT = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x08\x00\x00\x00\x08\x08\x06\x00\x00\x00\xc4\x0f\xbe\x8b"
    b"\x00\x00\x00\x06bKGD\x00\xff\x00\xff\x00\xff\xa0\xbd\xa7\x93\x00\x00\x00\x19tEXtSoftware\x00python.org"
    b"\x00\x00\x00\x14IDATx\x9cc\xf8\x0f\x04\x0c\x0c\x0c\x00\x00\x00\xff\xff\x03\x00\x01\x98\x00\x19g\xbb\r\x00"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _example_context():
    reg = get_registry()
    ctx = reg.minimal_context()
    ctx["title"] = "SpectraMind V50 – Example Diagnostics"
    # Populate some plausible values:
    ctx["metrics"].update(
        {
            "gll_mean": 0.1234,
            "rmse_mean": 0.0567,
            "calibration_error": 0.0123,
            "num_planets": 1100,
            "duration_seconds": 12345.6,
        }
    )
    ctx["run"].update(
        {
            "config_hash": "c0ffee" * 10,
            "git_commit": "deadbeefcafebabe",
            "cli": "spectramind diagnose dashboard --version",
            "host": "example-host",
        }
    )
    # Add a small base64 "heatmap" placeholder:
    ctx["artifacts"]["gll_heatmap_base64"] = base64.b64encode(_PNG_DOT).decode("ascii")
    # Example HTML fragments
    ctx["artifacts"][
        "umap_html"
    ] = "<div style='height:200px;border:1px dashed var(--border);border-radius:8px;display:flex;align-items:center;justify-content:center;'>UMAP HTML fragment</div>"
    ctx["artifacts"][
        "tsne_html"
    ] = "<div style='height:200px;border:1px dashed var(--border);border-radius:8px;display:flex;align-items:center;justify-content:center;'>t-SNE HTML fragment</div>"
    # Symbolic table rows
    ctx["artifacts"]["symbolic_rule_table_rows"] = [
        {
            "planet_id": "P0001",
            "rule_name": "H2O < 0 when CH4 high",
            "score": 0.87,
            "details": "bin[120:140]",
        },
        {
            "planet_id": "P0042",
            "rule_name": "FFT asymmetry > τ",
            "score": 0.73,
            "details": "FFT window 256",
        },
    ]
    # CLI log snippet
    ctx["cli_log_rows"] = [
        "[2025-08-15T10:10:10Z] spectramind diagnose dashboard --version V50",
        "[2025-08-15T10:15:10Z] spectramind analyze-log --group-by hash",
    ]
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=str(
            Path(__file__).resolve().parent.parent.parent.parent.parent / "_example_out"
        ),
    )
    ap.add_argument("--copy-assets", action="store_true", default=True)
    args = ap.parse_args()

    reg = get_registry()
    ctx = _example_context()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Full diagnostics HTML
    html_out = os.path.join(args.out_dir, "report.html")
    reg.render_to_file(
        "diagnostic_report.html.j2", ctx, html_out, copy_assets=args.copy_assets
    )

    # 2) Markdown summary
    md_out = os.path.join(args.out_dir, "summary.md")
    reg.render_to_file("diagnostic_summary.md.j2", ctx, md_out)

    # 3) Manifest JSON
    man_out = os.path.join(args.out_dir, "manifest.json")
    reg.render_to_file("manifest.json.j2", ctx, man_out)

    # 4) Bundle descriptor
    bundle_out = os.path.join(args.out_dir, "report_bundle.json")
    reg.render_to_file(
        "report_bundle.json.j2",
        {
            "generated_at_utc": ctx["generated_at_utc"],
            "output": {
                "report_html": "report.html",
                "summary_md": "summary.md",
                "manifest_json": "manifest.json",
            },
            "hashes": {
                "report_html": reg.content_hash(
                    open(html_out, "r", encoding="utf-8").read()
                ),
                "summary_md": reg.content_hash(
                    open(md_out, "r", encoding="utf-8").read()
                ),
                "manifest_json": reg.content_hash(
                    open(man_out, "r", encoding="utf-8").read()
                ),
            },
            "notes": "Example bundle produced by example_render.py",
        },
        bundle_out,
    )

    print(f"Wrote:\n- {html_out}\n- {md_out}\n- {man_out}\n- {bundle_out}")


if __name__ == "__main__":
    sys.exit(main())
