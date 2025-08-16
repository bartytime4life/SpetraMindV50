#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Calibration Schema Validator

Validates calibration configuration files (Hydra YAMLs or raw YAML dicts) against a canonical
schema and emits rich validation reports.  The tool is intentionally standalone so it can be
invoked either as a module (``python -m spectramind.calibration_schema``) or through the Typer
command line interface.

Key features
------------
* Recursive validator with JSONPath-like error paths.
* Enforcement for required fields, type checks, value ranges and custom validation rules.
* Optional rendering of the schema and example configurations.
* JSON/Markdown report generation and structured logging.

Exit codes
----------
0 = success (valid)
1 = validation errors present or I/O failure
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except Exception as e:  # pragma: no cover - import guard
    print("ERROR: PyYAML is required (pip install pyyaml).", file=sys.stderr)
    raise

try:
    import typer
except Exception:  # pragma: no cover - import guard
    typer = None  # type: ignore

try:  # optional pretty output
    from rich import print as rprint  # type: ignore
    from rich.console import Console  # type: ignore
    from rich.panel import Panel  # type: ignore
    from rich.table import Table  # type: ignore
    from rich.syntax import Syntax  # type: ignore
except Exception:  # pragma: no cover - rich is optional
    rprint = None
    Console = None
    Panel = None
    Table = None
    Syntax = None

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
APP_NAME = "SpectraMindV50-CalibrationSchema"
DEFAULT_SCHEMA_PATH = "configs/calibration/schema/default.yaml"
LOG_DIR = Path("logs/validator")
JSONL_LOG_PATH = LOG_DIR / "calibration_schema_events.jsonl"
ROTATING_LOG_PATH = LOG_DIR / "calibration_schema.log"
RUN_SEED = 42

# ---------------------------------------------------------------------------
# Logging Utilities
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def setup_logging() -> logging.Logger:
    _ensure_dir(LOG_DIR)
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers[:] = []

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    try:
        from logging.handlers import RotatingFileHandler

        fh = RotatingFileHandler(
            ROTATING_LOG_PATH, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(fh)
    except Exception:  # pragma: no cover - fallback
        fh = logging.FileHandler(ROTATING_LOG_PATH, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(fh)

    return logger

LOGGER = setup_logging()


def jsonl_event(event: Dict[str, Any]) -> None:
    """Append an event to the JSONL log."""
    try:
        _ensure_dir(JSONL_LOG_PATH.parent)
        event["ts"] = datetime.utcnow().isoformat() + "Z"
        with JSONL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover - logging failures are non fatal
        LOGGER.debug(f"Failed to write JSONL event: {e}")


def capture_env_git() -> Dict[str, Any]:
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cwd": str(Path.cwd()),
        "env_user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "git_commit": None,
        "git_branch": None,
    }
    try:
        import subprocess

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        info["git_commit"] = commit
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["git_branch"] = branch
    except Exception:
        pass
    return info

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def type_name(x: Any) -> str:
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "str"
    if isinstance(x, dict):
        return "dict"
    if isinstance(x, list):
        return "list"
    return type(x).__name__

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

# ---------------------------------------------------------------------------
# Validation models
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    path: str
    message: str
    severity: str = "error"
    code: str = "SCHEMA_VIOLATION"

@dataclass
class ValidationResult:
    ok: bool
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "errors": [vars(e) for e in self.errors],
            "warnings": [vars(w) for w in self.warnings],
        }

# ---------------------------------------------------------------------------
# Schema Validator
# ---------------------------------------------------------------------------

class SchemaValidator:
    """Validate a config dict against a schema dict."""

    def __init__(self, schema: Dict[str, Any]):
        if not isinstance(schema, dict) or "calibration_schema" not in schema:
            raise ValueError("Invalid schema: missing 'calibration_schema' root.")
        root = schema["calibration_schema"]
        self.fields = root.get("fields", {})
        self.rules = root.get("validation_rules", [])

    # Public API ------------------------------------------------------------
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        errors: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []
        if not isinstance(config, dict):
            errors.append(ValidationIssue(path="$", message="Config must be a mapping (dict)."))
            return ValidationResult(ok=False, errors=errors, warnings=warnings)

        for field_name, spec in self.fields.items():
            self._validate_field(config, field_name, spec, f"$.{field_name}", errors, warnings)

        for k in config.keys():
            if k not in self.fields:
                warnings.append(
                    ValidationIssue(
                        path=f"$.{k}",
                        message="Field not defined in schema.",
                        severity="warning",
                        code="EXTRA_FIELD",
                    )
                )

        self._validate_rules(config, errors, warnings)
        return ValidationResult(ok=not errors, errors=errors, warnings=warnings)

    # Individual field validation -----------------------------------------
    def _validate_field(
        self,
        config: Dict[str, Any],
        name: str,
        spec: Dict[str, Any],
        path: str,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        required = bool(spec.get("required", False))
        if name not in config:
            if required:
                errors.append(ValidationIssue(path=path, message="Missing required field."))
            return

        value = config[name]
        declared_type = spec.get("type")
        if declared_type and not self._check_type(value, declared_type):
            errors.append(
                ValidationIssue(
                    path=path,
                    message=f"Type mismatch: expected {declared_type}, got {type_name(value)}.",
                )
            )
            return

        if "allowed_values" in spec:
            allowed = spec["allowed_values"]
            if isinstance(value, list):
                for i, v in enumerate(value):
                    if v not in allowed:
                        errors.append(
                            ValidationIssue(
                                path=f"{path}[{i}]",
                                message=f"Value '{v}' not in allowed set: {allowed}.",
                            )
                        )
            else:
                if value not in allowed:
                    errors.append(
                        ValidationIssue(
                            path=path,
                            message=f"Value '{value}' not in allowed set: {allowed}.",
                        )
                    )

        if "range" in spec and is_number(value):
            rng = spec["range"]
            if isinstance(rng, list) and len(rng) == 2:
                lo, hi = rng
                if (lo is not None and value < lo) or (hi is not None and value > hi):
                    errors.append(
                        ValidationIssue(path=path, message=f"Value {value} out of range [{lo}, {hi}].")
                    )

        if spec.get("type") == "list":
            item_type = spec.get("item_type")
            if item_type and isinstance(value, list):
                for i, v in enumerate(value):
                    if not self._check_type(v, item_type):
                        errors.append(
                            ValidationIssue(
                                path=f"{path}[{i}]",
                                message=f"List item type mismatch: expected {item_type}, got {type_name(v)}.",
                            )
                        )

        if spec.get("type") == "dict" and "schema" in spec and isinstance(value, dict):
            nested_spec = spec["schema"]
            for sub_name, sub_spec in nested_spec.items():
                self._validate_field(
                    value, sub_name, sub_spec, f"{path}.{sub_name}", errors, warnings
                )
            for kk in value.keys():
                if kk not in nested_spec:
                    warnings.append(
                        ValidationIssue(
                            path=f"{path}.{kk}",
                            message="Field not defined in nested schema.",
                            severity="warning",
                            code="EXTRA_FIELD",
                        )
                    )

    def _check_type(self, value: Any, t: str) -> bool:
        if t == "str":
            return isinstance(value, str)
        if t == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        if t == "float":
            return is_number(value)
        if t == "bool":
            return isinstance(value, bool)
        if t == "list":
            return isinstance(value, list)
        if t == "dict":
            return isinstance(value, dict)
        return False

    # Rule evaluation ------------------------------------------------------
    def _validate_rules(
        self, config: Dict[str, Any], errors: List[ValidationIssue], warnings: List[ValidationIssue]
    ) -> None:
        for rule in self.rules:
            expr = rule.get("rule")
            msg = rule.get("error", "Validation rule violated.")
            if not expr or not isinstance(expr, str):
                continue
            try:
                if expr.strip().lower().startswith("if "):
                    m = re.match(r"if (.+?) then (.+)$", expr.strip(), re.IGNORECASE)
                    if not m:
                        errors.append(ValidationIssue(path="$", message=f"Invalid rule syntax: {expr}"))
                        continue
                    cond_if = m.group(1).strip()
                    cond_then = m.group(2).strip()
                    if self._eval_condition(cond_if, config):
                        if not self._eval_condition(cond_then, config):
                            errors.append(ValidationIssue(path="$", message=msg))
                else:
                    if not self._eval_condition(expr.strip(), config):
                        errors.append(ValidationIssue(path="$", message=msg))
            except Exception as e:
                errors.append(ValidationIssue(path="$", message=f"Error evaluating rule '{expr}': {e}"))

    def _eval_condition(self, cond: str, config: Dict[str, Any]) -> bool:
        ops = ["==", "!=", ">=", "<=", ">", "<"]
        op_regex = "(" + "|".join(map(re.escape, ops)) + ")"
        m = re.split(op_regex, cond)
        if len(m) != 3:
            path = cond.strip()
            val, exists = self._resolve_path(path, config)
            return bool(val) if exists else False
        lhs_raw, op, rhs_raw = m[0].strip(), m[1], m[2].strip()
        lhs_val, lhs_exists = self._resolve_path(lhs_raw, config)
        if not lhs_exists:
            return False
        rhs_val = self._parse_rhs(rhs_raw)
        if is_number(lhs_val) and is_number(rhs_val):
            a, b = float(lhs_val), float(rhs_val)
            if op == "==":
                return a == b
            if op == "!=":
                return a != b
            if op == ">":
                return a > b
            if op == ">=":
                return a >= b
            if op == "<":
                return a < b
            if op == "<=":
                return a <= b
        if op == "==":
            return lhs_val == rhs_val
        if op == "!=":
            return lhs_val != rhs_val
        raise ValueError(f"Invalid non-numeric comparison: {lhs_val} {op} {rhs_val}")

    def _resolve_path(self, path: str, config: Dict[str, Any]) -> Tuple[Any, bool]:
        if path in ("true", "false"):
            return (path == "true"), True
        if (path.startswith("\"") and path.endswith("\"")) or (
            path.startswith("'") and path.endswith("'")
        ):
            return path[1:-1], True
        cur: Any = config
        parts = path.split(".") if path else []
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None, False
        return cur, True

    def _parse_rhs(self, s: str) -> Any:
        if s.lower() == "true":
            return True
        if s.lower() == "false":
            return False
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except Exception:
            pass
        if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_result_console(
    result: ValidationResult,
    config_path: Optional[str] = None,
    schema_path: Optional[str] = None,
) -> None:
    title = f"[{APP_NAME}] Validation {'PASSED' if result.ok else 'FAILED'}"
    meta = f"config={config_path or '<dict>'} | schema={schema_path or DEFAULT_SCHEMA_PATH}"
    if Console and Table and Panel:
        console = Console()
        console.print(Panel(meta, title=title, expand=False))
        if result.errors:
            t = Table(title=f"Errors ({len(result.errors)})", show_lines=False)
            t.add_column("Path", style="bold red")
            t.add_column("Message", style="white")
            for e in result.errors:
                t.add_row(e.path, e.message)
            console.print(t)
        if result.warnings:
            t = Table(title=f"Warnings ({len(result.warnings)})", show_lines=False)
            t.add_column("Path", style="bold yellow")
            t.add_column("Message", style="white")
            for w in result.warnings:
                t.add_row(w.path, w.message)
            console.print(t)
    else:
        print(title)
        print(meta)
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for e in result.errors:
                print(f"  - {e.path}: {e.message}")
        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w.path}: {w.message}")

def export_report(
    result: ValidationResult,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
    config_path: Optional[str] = None,
    schema_path: Optional[str] = None,
) -> None:
    payload = {
        "app": APP_NAME,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "ok": result.ok,
        "config_path": config_path,
        "schema_path": schema_path,
        **result.to_dict(),
    }
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    if out_md:
        lines: List[str] = []
        lines.append(f"# {APP_NAME} — Validation {'PASSED' if result.ok else 'FAILED'}")
        lines.append("")
        lines.append(f"- Time (UTC): {payload['timestamp_utc']}")
        lines.append(f"- Config: `{config_path}`")
        lines.append(f"- Schema: `{schema_path}`")
        lines.append(f"- Errors: {payload['num_errors']}")
        lines.append(f"- Warnings: {payload['num_warnings']}")
        lines.append("")
        if payload["errors"]:
            lines.append("## Errors")
            for e in payload["errors"]:
                lines.append(f"- **{e['path']}** — {e['message']}")
        if payload["warnings"]:
            lines.append("")
            lines.append("## Warnings")
            for w in payload["warnings"]:
                lines.append(f"- **{w['path']}** — {w['message']}")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

if typer is not None:
    app = typer.Typer(add_completion=False, help="SpectraMind V50 Calibration Schema Validator")
else:  # pragma: no cover - fallback when typer missing
    class _DummyTyper:
        def command(self, *args: Any, **kwargs: Any):  # type: ignore
            def decorator(func):
                return func

            return decorator

        def __call__(self, *args: Any, **kwargs: Any) -> None:
            print("Typer is required for CLI usage (pip install typer).", file=sys.stderr)
            raise RuntimeError("typer not installed")

    app = _DummyTyper()  # type: ignore

def _load_schema(schema_path: Optional[str]) -> Dict[str, Any]:
    path = schema_path or DEFAULT_SCHEMA_PATH
    schema = load_yaml(path)
    if not isinstance(schema, dict):
        raise ValueError("Schema YAML did not parse to a mapping.")
    return schema

def _load_config(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML did not parse to a mapping.")
    return cfg

def _log_run_start(cmd: str, extras: Dict[str, Any]) -> str:
    run_id = str(uuid.uuid4())
    evt = {
        "event": "run_start",
        "run_id": run_id,
        "cmd": cmd,
        "extras": extras,
        **capture_env_git(),
    }
    jsonl_event(evt)
    LOGGER.info(f"Run {cmd} start | run_id={run_id}")
    return run_id

def _log_run_end(run_id: str, status: str, details: Dict[str, Any]) -> None:
    evt = {"event": "run_end", "run_id": run_id, "status": status, "details": details}
    jsonl_event(evt)
    LOGGER.info(f"Run end | run_id={run_id} | status={status}")

@app.command("validate")
def cli_validate(
    config: str = typer.Option(..., "--config", "-c", help="Path to calibration config YAML to validate."),
    schema: Optional[str] = typer.Option(
        None, "--schema", "-s", help=f"Path to schema YAML (default: {DEFAULT_SCHEMA_PATH})."
    ),
    out_json: Optional[str] = typer.Option(None, "--out-json", help="Optional path to save JSON validation report."),
    out_md: Optional[str] = typer.Option(None, "--out-md", help="Optional path to save Markdown validation report."),
) -> None:
    """Validate a calibration config YAML against the schema."""
    run_id = _log_run_start(
        "validate", {"config": config, "schema": schema, "out_json": out_json, "out_md": out_md}
    )
    try:
        schema_dict = _load_schema(schema)
        config_dict = _load_config(config)
        validator = SchemaValidator(schema_dict)
        result = validator.validate(config_dict)
        render_result_console(result, config_path=config, schema_path=(schema or DEFAULT_SCHEMA_PATH))
        export_report(
            result,
            out_json=out_json,
            out_md=out_md,
            config_path=config,
            schema_path=(schema or DEFAULT_SCHEMA_PATH),
        )
        status = "success" if result.ok else "failed"
        _log_run_end(run_id, status, {"num_errors": len(result.errors), "num_warnings": len(result.warnings)})
        sys.exit(0 if result.ok else 1)
    except Exception as e:
        tb = traceback.format_exc()
        LOGGER.error(f"Validation exception: {e}")
        _log_run_end(run_id, "error", {"exception": str(e), "traceback": tb})
        print(tb, file=sys.stderr)
        sys.exit(1)

@app.command("print-schema")
def cli_print_schema(
    schema: Optional[str] = typer.Option(
        None, "--schema", "-s", help=f"Path to schema YAML (default: {DEFAULT_SCHEMA_PATH})."
    )
) -> None:
    """Pretty-print the active calibration schema."""
    run_id = _log_run_start("print-schema", {"schema": schema})
    try:
        schema_dict = _load_schema(schema)
        if rprint and Syntax:
            src = yaml.safe_dump(schema_dict, sort_keys=False, allow_unicode=True)
            rprint(Syntax(src, "yaml", theme="monokai", word_wrap=True))
        else:
            print(yaml.safe_dump(schema_dict, sort_keys=False, allow_unicode=True))
        _log_run_end(run_id, "success", {})
    except Exception as e:
        tb = traceback.format_exc()
        LOGGER.error(f"print-schema exception: {e}")
        _log_run_end(run_id, "error", {"exception": str(e), "traceback": tb})
        print(tb, file=sys.stderr)
        sys.exit(1)

@app.command("examples")
def cli_examples(
    schema: Optional[str] = typer.Option(
        None, "--schema", "-s", help=f"Path to schema YAML (default: {DEFAULT_SCHEMA_PATH})."
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Filter by example name substring."),
) -> None:
    """Show example configs embedded in the schema."""
    run_id = _log_run_start("examples", {"schema": schema, "name": name})
    try:
        schema_dict = _load_schema(schema)
        calib = schema_dict.get("calibration_schema", {})
        examples = calib.get("examples", [])
        if not examples:
            print("No examples defined in schema.")
            _log_run_end(run_id, "success", {"num_examples": 0})
            return
        for ex in examples:
            ename = ex.get("name", "<unnamed>")
            if name and name.lower() not in ename.lower():
                continue
            cfg = ex.get("config", {})
            print("=" * 80)
            print(f"Example: {ename}")
            print("-" * 80)
            print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        _log_run_end(run_id, "success", {"num_examples": len(examples)})
    except Exception as e:
        tb = traceback.format_exc()
        LOGGER.error(f"examples exception: {e}")
        _log_run_end(run_id, "error", {"exception": str(e), "traceback": tb})
        print(tb, file=sys.stderr)
        sys.exit(1)

@app.command("gen-report")
def cli_gen_report(
    config: str = typer.Option(..., "--config", "-c", help="Path to calibration config YAML to validate."),
    schema: Optional[str] = typer.Option(
        None, "--schema", "-s", help=f"Path to schema YAML (default: {DEFAULT_SCHEMA_PATH})."
    ),
    out_json: str = typer.Option("validation_report.json", "--out-json", help="Path to write JSON report."),
    out_md: Optional[str] = typer.Option("validation_report.md", "--out-md", help="Path to write Markdown report."),
) -> None:
    """Validate and export a report (JSON and/or Markdown)."""
    run_id = _log_run_start(
        "gen-report", {"config": config, "schema": schema, "out_json": out_json, "out_md": out_md}
    )
    try:
        schema_dict = _load_schema(schema)
        config_dict = _load_config(config)
        validator = SchemaValidator(schema_dict)
        result = validator.validate(config_dict)
        render_result_console(result, config_path=config, schema_path=(schema or DEFAULT_SCHEMA_PATH))
        export_report(
            result,
            out_json=out_json,
            out_md=out_md,
            config_path=config,
            schema_path=(schema or DEFAULT_SCHEMA_PATH),
        )
        status = "success" if result.ok else "failed"
        _log_run_end(run_id, status, {"num_errors": len(result.errors), "num_warnings": len(result.warnings)})
        sys.exit(0 if result.ok else 1)
    except Exception as e:
        tb = traceback.format_exc()
        LOGGER.error(f"gen-report exception: {e}")
        _log_run_end(run_id, "error", {"exception": str(e), "traceback": tb})
        print(tb, file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover - simple delegator
    try:
        import random

        random.seed(RUN_SEED)
    except Exception:
        pass
    app()

if __name__ == "__main__":  # pragma: no cover
    main()
