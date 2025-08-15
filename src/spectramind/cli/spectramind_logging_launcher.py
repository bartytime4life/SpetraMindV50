"""
Drop-in launcher that wraps an existing Typer root app with logging instrumentation.
It attempts to import a root CLI callable from common locations and names.

Run as:
python -m spectramind.cli.spectramind_logging_launcher -- spectramind.spectramind:app train --config configs/config_v50.yaml

Where the first argument after '--' is a module:attr path to the Typer app.
Any subsequent args are passed to the underlying CLI.
"""

import importlib
import sys
from typing import Any, Dict, Tuple

from spectramind.cli._logging_instrumentation import instrument_typer_app
from spectramind.logging import get_logger


def _parse_target(argv) -> Tuple[str, str, list]:
    if "--" in argv:
        dash = argv.index("--")
        rest = argv[dash + 1 :]
    else:
        rest = argv

    if not rest:
        raise SystemExit("Usage: ... -- <module_path>:<attr_name> [cli args...]")
    target, *cli_args = rest
    if ":" not in target:
        raise SystemExit(
            "Target must be of form module:attr (e.g., spectramind.spectramind:app)"
        )
    mpath, attr = target.split(":", 1)
    return mpath, attr, cli_args


def _collect_args_for_log(cli_args: list) -> Dict[str, Any]:
    # naive parse for logging: --key value or --flag
    args = {}
    i = 0
    while i < len(cli_args):
        tok = cli_args[i]
        if tok.startswith("--"):
            if i + 1 < len(cli_args) and not cli_args[i + 1].startswith("-"):
                args[tok] = cli_args[i + 1]
                i += 2
            else:
                args[tok] = True
                i += 1
        else:
            i += 1
    return args


def main():
    get_logger("spectramind.launcher")
    mpath, attr, cli_args = _parse_target(sys.argv[1:])
    args_for_log = _collect_args_for_log(cli_args)
    mod = importlib.import_module(mpath)
    app = getattr(mod, attr, None)
    if app is None:
        raise SystemExit(f"Could not find attribute '{attr}' in module '{mpath}'")
    ctx, _meta = instrument_typer_app(
        app, command_name=f"{mpath}:{attr}", args=args_for_log
    )
    with ctx:
        # hand off to Typer app by reconstructing argv as if executed directly
        sys.argv = [f"{mpath}:{attr}"] + cli_args
        app()


if __name__ == "__main__":
    main()
