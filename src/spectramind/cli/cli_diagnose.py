from pathlib import Path
from typing import Optional

import typer

from .common import (
    PROJECT_ROOT,
    SRC_DIR,
    REPORTS_DIR,
    logger,
    ensure_tools,
    command_session,
    find_module_or_script,
    call_python_module,
    call_python_file,
    open_in_browser,
)

app = typer.Typer(no_args_is_help=True, help="SpectraMind V50 — Diagnostics & Dashboard")


@app.command("dashboard")
def dashboard(
    preds_dir: str = typer.Option("predictions", "--preds", help="Predictions directory"),
    outdir: str = typer.Option("diagnostics", "--outdir", help="Diagnostics output directory"),
    html_out: str = typer.Option("reports/diagnostic_report_v1.html", "--html-out"),
    open_browser: bool = typer.Option(False, "--open-browser/--no-open-browser"),
    no_umap: bool = typer.Option(False, "--no-umap", help="Skip UMAP render"),
    no_tsne: bool = typer.Option(False, "--no-tsne", help="Skip t-SNE render"),
):
    """Build unified diagnostics: GLL heatmap, UMAP/t-SNE, SHAP overlays, symbolic rule leaderboard,
    COREL calibration plots, and an interactive HTML dashboard."""
    ensure_tools()
    args_summary = ["--preds", preds_dir, "--outdir", outdir, "--emit-json"]
    with command_session("diagnose.dashboard", ["--preds", preds_dir, "--outdir", outdir, "--html", html_out]):
        module_sum = "spectramind.generate_diagnostic_summary"
        cand_sum = [SRC_DIR / "spectramind" / "generate_diagnostic_summary.py"]
        k, s = find_module_or_script(module_sum, cand_sum)
        if k == "module":
            rc = call_python_module(module_sum, args_summary)
        elif k == "script" and s:
            rc = call_python_file(s, args_summary)
        else:
            logger.error("Missing generate_diagnostic_summary.")
            raise typer.Exit(2)
        if rc != 0:
            raise typer.Exit(rc)
        module_html = "spectramind.generate_html_report"
        cand_html = [SRC_DIR / "spectramind" / "generate_html_report.py"]
        args_html = ["--preds", preds_dir, "--diagnostics", outdir, "--html-out", html_out]
        if no_umap:
            args_html.append("--no-umap")
        if no_tsne:
            args_html.append("--no-tsne")
        k2, s2 = find_module_or_script(module_html, cand_html)
        if k2 == "module":
            rc2 = call_python_module(module_html, args_html)
        elif k2 == "script" and s2:
            rc2 = call_python_file(s2, args_html)
        else:
            logger.error("Missing generate_html_report.")
            raise typer.Exit(2)
        if rc2 == 0 and open_browser:
            open_in_browser(html_out)
        raise typer.Exit(rc2)


@app.command("gll-heatmap")
def gll_heatmap(
    preds_dir: str = typer.Option("predictions", "--preds", help="Predictions directory"),
    outdir: str = typer.Option("diagnostics", "--outdir", help="Diagnostics output directory"),
):
    """Render bin-wise GLL heatmap and summary."""
    ensure_tools()
    with command_session("diagnose.gll-heatmap", ["--preds", preds_dir, "--outdir", outdir]):
        module = "spectramind.plot_gll_heatmap_per_bin"
        candidates = [SRC_DIR / "spectramind" / "plot_gll_heatmap_per_bin.py"]
        k, s = find_module_or_script(module, candidates)
        if k == "module":
            rc = call_python_module(module, ["--preds", preds_dir, "--outdir", outdir])
        elif k == "script" and s:
            rc = call_python_file(s, ["--preds", preds_dir, "--outdir", outdir])
        else:
            logger.error("Missing plot_gll_heatmap_per_bin.")
            raise typer.Exit(2)
        raise typer.Exit(rc)
