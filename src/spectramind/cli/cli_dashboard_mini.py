from typing import Optional

import typer

from .common import (
    SRC_DIR,
    logger,
    ensure_tools,
    command_session,
    find_module_or_script,
    call_python_module,
    call_python_file,
    open_in_browser,
)

app = typer.Typer(no_args_is_help=True, help="SpectraMind V50 — Mini dashboard runner")


@app.command("run")
def run(
    preds_dir: str = typer.Option("predictions", "--preds"),
    out_html: str = typer.Option("reports/mini_dashboard.html", "--html"),
    open_browser: bool = typer.Option(False, "--open-browser/--no-open-browser"),
):
    """Fast-path: generate diagnostic summary + HTML report in one go, optimized defaults."""
    ensure_tools()
    args = ["--preds", preds_dir, "--out", out_html]
    with command_session("dashboard.mini", args):
        module_html = "spectramind.generate_html_report"
        cand_html = [SRC_DIR / "spectramind" / "generate_html_report.py"]
        k, s = find_module_or_script(module_html, cand_html)
        if k == "module":
            rc = call_python_module(module_html, ["--quick", "--preds", preds_dir, "--html-out", out_html])
        elif k == "script" and s:
            rc = call_python_file(s, ["--quick", "--preds", preds_dir, "--html-out", out_html])
        else:
            logger.error("Missing generate_html_report.")
            raise typer.Exit(2)
        if rc == 0 and open_browser:
            open_in_browser(out_html)
        raise typer.Exit(rc)
