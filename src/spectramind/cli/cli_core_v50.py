from __future__ import annotations

"""
SpectraMind V50 — Core CLI (training, inference, calibration hooks) with runtime wiring.

Every command resolves a runtime (CLI param -> ENV -> default) and composes a Hydra config
that includes the override runtime=<name>. The composed config is printed on demand and
passed into the underlying implementation.

This file intentionally avoids heavy imports at module load time to keep CLI latency low.
"""

from typing import Optional, List

import typer

from spectramind.conf_helpers.runtime import (
    determine_runtime,
    set_runtime_env,
    compose_with_runtime,
    log_runtime_choice,
    pretty_print_cfg,
)


app = typer.Typer(help="SpectraMind V50 — Core CLI (training / inference)")


@app.command("train")
def train(
    runtime: Optional[str] = typer.Option(
        None, "--runtime", "-r", help="Execution environment (Hydra group)"
    ),
    config_overrides: List[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Additional Hydra overrides (repeatable), e.g. -o model=v50/fgs1_mamba -o data.batch=8",
        rich_help_panel="Hydra",
    ),
    show_cfg: bool = typer.Option(False, "--show-cfg", help="Print resolved Hydra config and exit"),
):
    """
    Train the SpectraMind V50 model stack. Respects the chosen runtime.
    """
    resolved_runtime = determine_runtime(runtime)
    set_runtime_env(resolved_runtime)
    cfg = compose_with_runtime(resolved_runtime, overrides=config_overrides or [])
    log_runtime_choice(resolved_runtime, cfg, extra_note="core.train")

    if show_cfg:
        pretty_print_cfg(cfg)
        raise typer.Exit(0)

    # Defer heavy imports until we actually run training.
    try:
        # Expect an implementation function: train_v50.train_from_config(cfg)
        from spectramind.models.train_v50 import train_from_config

        exit_code = train_from_config(cfg)  # type: ignore[call-arg]
    except ModuleNotFoundError:
        # Fallback: if training module isn't present, fail gracefully with guidance.
        typer.echo(
            "[ERROR] Module 'spectramind.models.train_v50' not found. "
            "Please ensure your repository includes the V50 training implementation."
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"[ERROR] Training failed: {e}")
        raise typer.Exit(1)

    raise typer.Exit(exit_code if isinstance(exit_code, int) else 0)


@app.command("predict")
def predict(
    runtime: Optional[str] = typer.Option(
        None, "--runtime", "-r", help="Execution environment (Hydra group)"
    ),
    config_overrides: List[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Additional Hydra overrides for inference",
        rich_help_panel="Hydra",
    ),
    show_cfg: bool = typer.Option(False, "--show-cfg", help="Print resolved Hydra config and exit"),
):
    """
    Run inference with the V50 pipeline (μ/σ generation, packaging optional).
    """
    resolved_runtime = determine_runtime(runtime)
    set_runtime_env(resolved_runtime)
    cfg = compose_with_runtime(resolved_runtime, overrides=config_overrides or [])
    log_runtime_choice(resolved_runtime, cfg, extra_note="core.predict")

    if show_cfg:
        pretty_print_cfg(cfg)
        raise typer.Exit(0)

    try:
        from spectramind.infer.predict_v50 import predict_from_config

        exit_code = predict_from_config(cfg)  # type: ignore[call-arg]
    except ModuleNotFoundError:
        typer.echo(
            "[ERROR] Module 'spectramind.infer.predict_v50' not found. "
            "Please ensure your repository includes the V50 inference implementation."
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"[ERROR] Prediction failed: {e}")
        raise typer.Exit(1)

    raise typer.Exit(exit_code if isinstance(exit_code, int) else 0)


@app.command("calibrate")
def calibrate(
    runtime: Optional[str] = typer.Option(
        None, "--runtime", "-r", help="Execution environment (Hydra group)"
    ),
    config_overrides: List[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Additional Hydra overrides for calibration",
        rich_help_panel="Hydra",
    ),
    show_cfg: bool = typer.Option(False, "--show-cfg", help="Print resolved Hydra config and exit"),
):
    """
    Run uncertainty calibration (e.g., temperature scaling + COREL).
    """
    resolved_runtime = determine_runtime(runtime)
    set_runtime_env(resolved_runtime)
    cfg = compose_with_runtime(resolved_runtime, overrides=config_overrides or [])
    log_runtime_choice(resolved_runtime, cfg, extra_note="core.calibrate")

    if show_cfg:
        pretty_print_cfg(cfg)
        raise typer.Exit(0)

    try:
        from spectramind.calibration.calibrate_v50 import calibrate_from_config

        exit_code = calibrate_from_config(cfg)  # type: ignore[call-arg]
    except ModuleNotFoundError:
        typer.echo(
            "[ERROR] Module 'spectramind.calibration.calibrate_v50' not found. "
            "Please ensure your repository includes the V50 calibration implementation."
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"[ERROR] Calibration failed: {e}")
        raise typer.Exit(1)

    raise typer.Exit(exit_code if isinstance(exit_code, int) else 0)

