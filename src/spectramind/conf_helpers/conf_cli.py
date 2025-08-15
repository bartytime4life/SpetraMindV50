# SPDX-License-Identifier: MIT

"""Typer-based CLI for configuration utilities."""

import json
from pathlib import Path
from typing import Optional, List

import typer
from omegaconf import OmegaConf

from .logging_utils import init_logging, write_md, log_event
from .hydra_integration import load_config_hydra
from .loader import load_and_validate
from .overrides import load_overrides_layered
from .validators import validate_config
from .schema import get_json_schema
from .hashing import config_hash
from .io import save_json, save_yaml

app = typer.Typer(add_completion=False, help="SpectraMind V50 config helpers CLI")


@app.callback()
def _bootstrap(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    init_logging()


@app.command("print")
def cmd_print(
    config_dir: Path = typer.Argument(..., exists=True, readable=True),
    config_name: str = typer.Option("config_v50.yaml", help="Config file name"),
    override: Optional[List[str]] = typer.Option(None, help="Hydra override string(s)"),
    layered_override: Optional[List[Path]] = typer.Option(
        None, help="Apply multiple YAML overrides in order"
    ),
    symbolic_override: Optional[Path] = typer.Option(
        None, help="Apply one YAML override at the end"
    ),
    as_yaml: bool = typer.Option(True, help="Print YAML (else JSON)"),
) -> None:
    cfg = load_and_validate(
        config_dir=config_dir,
        config_name=config_name,
        overrides=override,
        layered_override_paths=layered_override,
        symbolic_override_path=symbolic_override,
    )
    if as_yaml:
        typer.echo(OmegaConf.to_yaml(cfg))
    else:
        typer.echo(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))


@app.command("validate")
def cmd_validate(
    config_dir: Path = typer.Argument(..., exists=True, readable=True),
    config_name: str = typer.Option("config_v50.yaml", help="Config file name"),
    override: Optional[List[str]] = typer.Option(None, help="Hydra override string(s)"),
) -> None:
    cfg = load_config_hydra(config_dir, config_name, override)
    validate_config(cfg)
    typer.echo("OK")


@app.command("hash")
def cmd_hash(
    config_dir: Path = typer.Argument(..., exists=True, readable=True),
    config_name: str = typer.Option("config_v50.yaml", help="Config file name"),
    override: Optional[List[str]] = typer.Option(None, help="Hydra override string(s)"),
) -> None:
    cfg = load_config_hydra(config_dir, config_name, override)
    h = config_hash(cfg)
    typer.echo(h)


@app.command("apply-overrides")
def cmd_apply_overrides(
    input_yaml: Path = typer.Argument(..., exists=True, readable=True),
    overrides: List[Path] = typer.Argument(..., exists=True, readable=True),
    out_yaml: Path = typer.Option(Path("merged_config.yaml"), help="Write merged YAML here"),
) -> None:
    base = OmegaConf.create(json.load(open(input_yaml, "r", encoding="utf-8")))
    merged = load_overrides_layered(base, overrides)
    save_yaml(OmegaConf.to_container(merged, resolve=True), out_yaml)
    typer.echo(str(out_yaml))


@app.command("dump-schema")
def cmd_dump_schema(out_json: Path = typer.Argument(Path("v50_config.schema.json"))) -> None:
    schema = get_json_schema()
    save_json(schema, out_json)
    typer.echo(str(out_json))


@app.command("log-state")
def cmd_log_state(
    note: str = typer.Argument(..., help="Message to log into v50_debug_log.md"),
) -> None:
    write_md(note, {})
    log_event("user_log", {"note": note})
    typer.echo("logged")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
