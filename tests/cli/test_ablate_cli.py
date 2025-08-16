# -*- coding: utf-8 -*-
"""Tests for the ablate CLI module."""
import importlib

from typer.testing import CliRunner


def test_ablate_help(runner: CliRunner, env_test_mode, repo_tmp_files):
    app_mod = importlib.import_module("spectramind.cli.cli_ablate")
    app = getattr(app_mod, "app", None)
    assert app is not None

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, f"ablate --help failed: {result.stdout}\n{result.stderr}"
    for token in ["--top-n", "--md", "--open-html"]:
        assert token in result.stdout or token.replace("-", "_") in result.stdout
