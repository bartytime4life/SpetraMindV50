# -*- coding: utf-8 -*-
"""Tests for the submit CLI module."""
import importlib

from typer.testing import CliRunner


def test_submit_help(runner: CliRunner, env_test_mode, repo_tmp_files):
    app_mod = importlib.import_module("spectramind.cli.cli_submit")
    app = getattr(app_mod, "app", None)
    assert app is not None

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, f"submit --help failed: {result.stdout}\n{result.stderr}"
    assert "submission" in result.stdout.lower()
