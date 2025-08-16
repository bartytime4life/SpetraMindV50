# -*- coding: utf-8 -*-
"""Tests for the core-v50 CLI module."""
import importlib

from typer.testing import CliRunner


def test_core_v50_help(runner: CliRunner, env_test_mode, repo_tmp_files):
    app_mod = importlib.import_module("spectramind.cli.cli_core_v50")
    app = getattr(app_mod, "app", None)
    assert app is not None

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, f"core-v50 --help failed: {result.stdout}\n{result.stderr}"
    for token in ["train", "predict", "calibrate"]:
        assert token in result.stdout
