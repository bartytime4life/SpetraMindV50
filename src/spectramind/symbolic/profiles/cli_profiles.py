"""Typer CLI for Symbolic Profiles."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from .logging_utils import (
    append_v50_debug_log,
    setup_logging,
    write_jsonl_event,
)
from .profile_diagnostics import generate_heatmap
from .profile_loader import (
    load_profiles_with_overrides,
    validate_profile_dict,
)
from .profile_registry import get_registry
from .utils import find_repo_root, safe_dump_yaml, safe_load_yaml

app = typer.Typer(add_completion=True, no_args_is_help=True, help="SpectraMind V50 — Symbolic Profiles CLI")


def _version_line(repo_root: Path, combined_hash: str) -> str:
    return f"SpectraMind V50 | profiles-cli | hash={combined_hash}"


@app.callback()
def _main(
    ctx: typer.Context,
    repo_root: Optional[Path] = typer.Option(None, "--repo-root", help="Override repo root"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose console logging"),
) -> None:
    root = find_repo_root(repo_root)
    logger = setup_logging("profiles.cli", repo_root=root)
    ctx.obj = {"root": root, "logger": logger, "verbose": verbose}


@app.command("list")
def cmd_list(
    examples: Optional[List[Path]] = typer.Option(
        None, "--examples", help="Optional example dirs/files to include (can pass multiple)."
    ),
) -> None:
    """List available profile IDs after applying overrides and optional examples."""
    ctx = typer.get_current_context()
    root: Path = ctx.obj["root"]
    res = load_profiles_with_overrides(repo_root=root, extra_sources=examples)
    reg = get_registry(root)
    reg.load_from_result(res)

    write_jsonl_event(root, {"event": "profiles_list", "ids": reg.list_ids(), "hash": res.combined_hash})
    append_v50_debug_log(root, _version_line(root, res.combined_hash))

    for pid in reg.list_ids():
        h = reg.get_hash(pid) or ""
        typer.echo(f"{pid}\t{h}")


@app.command("show")
def cmd_show(
    profile_id: str = typer.Argument(..., help="Profile ID to show"),
    examples: Optional[List[Path]] = typer.Option(None, "--examples", help="Example dirs/files to include."),
) -> None:
    """Show a profile YAML after merge/overrides."""
    ctx = typer.get_current_context()
    root: Path = ctx.obj["root"]
    res = load_profiles_with_overrides(repo_root=root, extra_sources=examples)
    reg = get_registry(root)
    reg.load_from_result(res)

    p = reg.get(profile_id)
    if not p:
        raise typer.Exit(code=2)
    payload = {
        "id": p.id,
        "name": p.name,
        "description": p.description,
        "tags": p.tags,
        "metadata": p.metadata,
        "rules": [r.__dict__ for r in p.rules],
        "hash": reg.get_hash(profile_id),
        "sources": [str(s) for s in res.source_maps.get(profile_id, [])],
    }
    out = safe_dump_yaml({"profile": payload})
    typer.echo(out)


@app.command("validate")
def cmd_validate(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Profile YAML file to validate"),
) -> None:
    """Validate a profile YAML against our lightweight schema."""
    ctx = typer.get_current_context()
    root: Path = ctx.obj["root"]
    if file:
        doc = safe_load_yaml(file)
        validate_profile_dict(doc)
        typer.echo(f"OK: {file}")
    else:
        res = load_profiles_with_overrides(repo_root=root)
        typer.echo(f"OK: {len(res.profiles)} profiles | combined hash={res.combined_hash}")


@app.command("diagnose")
def cmd_diagnose(
    violations_json: Path = typer.Option(..., "--viol", help="Violations JSON path"),
    out_dir: Path = typer.Option(Path("reports/profiles"), "--out-dir", help="Output directory"),
    metric: str = typer.Option("mean", "--metric", help="Metric for heatmap cell value [count|sum|mean]"),
    key_hint: Optional[str] = typer.Option(None, "--key-hint", help="Key to extract violation list"),
) -> None:
    """Generate Profile×Rule heatmap CSV and JSON summary from violations JSON."""
    ctx = typer.get_current_context()
    root: Path = ctx.obj["root"]
    out_dir = (root / out_dir).resolve()
    out_csv = out_dir / "profile_rule_heatmap.csv"
    out_json = out_dir / "profile_rule_summary.json"

    generate_heatmap(violations_json.resolve(), out_csv, out_json, key_hint=key_hint, metric=metric)

    write_jsonl_event(root, {"event": "profiles_diagnose", "violations": str(violations_json), "out": str(out_dir)})
    append_v50_debug_log(root, "profiles-cli diagnose completed")


@app.command("activate")
def cmd_activate(profile_id: str = typer.Argument(..., help="Profile ID to set active")) -> None:
    """Persist the active profile to runtime/active_profile.yaml."""
    ctx = typer.get_current_context()
    root: Path = ctx.obj["root"]
    res = load_profiles_with_overrides(repo_root=root)
    reg = get_registry(root)
    reg.load_from_result(res)

    if profile_id not in reg.list_ids():
        typer.echo(f"Profile '{profile_id}' not found", err=True)
        raise typer.Exit(code=2)

    reg.set_active(profile_id)
    write_jsonl_event(root, {"event": "profiles_activate", "id": profile_id})
    append_v50_debug_log(root, f"profiles-cli activate id={profile_id}")
    typer.echo(f"Active profile set: {profile_id}")


@app.command("export")
def cmd_export(
    profile_id: str = typer.Argument(..., help="Profile ID to export"),
    out_path: Path = typer.Option(..., "--out", help="Destination YAML path"),
) -> None:
    """Export a single, merged profile to YAML."""
    ctx = typer.get_current_context()
    root: Path = ctx.obj["root"]
    res = load_profiles_with_overrides(repo_root=root)
    reg = get_registry(root)
    reg.load_from_result(res)

    p = reg.get(profile_id)
    if not p:
        typer.echo(f"Profile '{profile_id}' not found", err=True)
        raise typer.Exit(code=2)

    out_path = out_path if out_path.is_absolute() else (root / out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profiles": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "tags": p.tags,
                "metadata": p.metadata,
                "rules": [r.__dict__ for r in p.rules],
            }
        ]
    }
    out_path.write_text(safe_dump_yaml(payload), encoding="utf-8")
    write_jsonl_event(root, {"event": "profiles_export", "id": profile_id, "out": str(out_path)})
    append_v50_debug_log(root, f"profiles-cli export id={profile_id} -> {out_path}")
    typer.echo(str(out_path))


if __name__ == "__main__":
    app()
