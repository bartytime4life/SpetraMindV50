# -*- coding: utf-8 -*-
"""Tests for the top-level SpectraMind CLI application."""
from typer.testing import CliRunner

from .test_cli_root import _import_cli


def test_root_cli_help(runner: CliRunner, env_test_mode, repo_tmp_files, tmp_path, monkeypatch):
    cli = _import_cli(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, f"--help failed: {result.stdout}\n{result.stderr}"
    for token in ["version", "test", "diagnose-log"]:
        assert token in result.stdout


def test_root_cli_version_command(runner: CliRunner, env_test_mode, repo_tmp_files, tmp_path, monkeypatch):
    cli = _import_cli(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert "cli_version" in result.stdout
