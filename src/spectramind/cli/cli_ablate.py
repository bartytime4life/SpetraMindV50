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
)

app = typer.Typer(no_args_is_help=True, help="SpectraMind V50 — Symbolic-aware ablation engine")


@app.command("run")
def run(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Base Hydra config"),
    top_n: int = typer.Option(5, "--top-n", help="Export top-N leaderboard"),
    md: bool = typer.Option(True, "--md/--no-md", help="Write Markdown leaderboard"),
    open_html: bool = typer.Option(False, "--open-html/--no-open-html", help="Open HTML leaderboard after run"),
):
    """Execute auto_ablate_v50 with full diagnostics, symbolic scoring, and leaderboard export."""
    ensure_tools()
    args: list[str] = []
    if config:
        args += ["--config", config]
    args += ["--top-n", str(top_n)]
    if md:
        args += ["--md"]
    if open_html:
        args += ["--open-html"]
    with command_session("ablate.run", args):
        module = "spectramind.auto_ablate_v50"
        candidates = [SRC_DIR / "spectramind" / "auto_ablate_v50.py"]
        k, s = find_module_or_script(module, candidates)
        if k == "module":
            rc = call_python_module(module, args)
        elif k == "script" and s:
            rc = call_python_file(s, args)
        else:
            logger.error("Missing auto_ablate_v50.")
            raise typer.Exit(2)
        raise typer.Exit(rc)
