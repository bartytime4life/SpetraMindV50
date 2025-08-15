import os
import json
import typer
import logging
from typing import Optional

from .report_generator import ReportGenerator, ReportConfig
from .report_data_collector import CollectorConfig
from .export_manager import ExportConfig

app = typer.Typer(help="SpectraMind V50 — Reporting CLI")


def _logger() -> logging.Logger:
    from .report_generator import _ensure_logging

    return _ensure_logging(log_name="v50_reporting_cli")


@app.command("generate")
def generate(
    diagnostics_dir: str = typer.Option("artifacts/diagnostics", help="Input diagnostics directory."),
    outputs_dir: str = typer.Option("reports", help="Where to write the report files."),
    version: str = typer.Option("v1", help="Version tag for output files."),
    title: str = typer.Option("SpectraMind V50 — Diagnostics Report", help="Report title."),
    subtitle: str = typer.Option("NeurIPS Ariel Data Challenge 2025", help="Report subtitle."),
    open_after: bool = typer.Option(False, help="Open the HTML after generation."),
    enable_pdf: bool = typer.Option(False, help="Attempt to export a PDF."),
    mlflow_log: bool = typer.Option(False, help="Log artifacts to MLflow if available."),
    wandb_log: bool = typer.Option(False, help="Log artifacts to Weights & Biases if available."),
):
    """Generate a diagnostics dashboard (HTML/MD, optional PDF)."""
    logger = _logger()
    rcfg = ReportConfig(
        diagnostics_dir=diagnostics_dir,
        outputs_dir=outputs_dir,
        report_version=version,
        title=title,
        subtitle=subtitle,
        open_after=open_after,
        enable_pdf=enable_pdf,
        mlflow_log=mlflow_log,
        wandb_log=wandb_log,
    )
    gen = ReportGenerator(
        report_cfg=rcfg,
        collector_cfg=CollectorConfig(),
        export_cfg=ExportConfig(),
        logger=logger,
    )
    paths = gen.render()
    typer.echo(json.dumps(paths, indent=2))


@app.command("open")
def open_report(
    html_path: str = typer.Argument(..., help="Path to an HTML report file."),
):
    """Open a previously generated HTML report in the default browser."""
    import webbrowser

    p = os.path.abspath(html_path)
    if not os.path.exists(p):
        raise typer.BadParameter(f"Not found: {p}")
    webbrowser.open(f"file://{p}")
    typer.echo(f"Opened: {p}")


@app.command("export-md")
def export_md(
    html_path: str = typer.Argument(..., help="An existing HTML report (to extract plain text)."),
    md_out: str = typer.Option("reports/extracted.md", help="Output Markdown path."),
):
    """Utility: crude HTML→Markdown extraction (best-effort for archiving)."""
    logger = _logger()
    try:
        import html2text  # optional

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        text = open(html_path, "r", encoding="utf-8").read()
        md = converter.handle(text)
    except Exception:
        # Fallback: strip tags minimally
        import re

        text = open(html_path, "r", encoding="utf-8").read()
        md = re.sub("<[^<]+?>", "", text)
    os.makedirs(os.path.dirname(md_out), exist_ok=True)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"Saved Markdown: {md_out}")
    typer.echo(md_out)


if __name__ == "__main__":
    app()
