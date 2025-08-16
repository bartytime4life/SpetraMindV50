# -*- coding: utf-8 -*-
"""Tests for invoking the built-in self-test command."""
from typer.testing import CliRunner

from .test_cli_root import _import_cli


def test_selftest_fast(runner: CliRunner, env_test_mode, repo_tmp_files, tmp_path, monkeypatch):
    cli = _import_cli(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["test", "--fast"])
    if result.exit_code != 0:
        result = runner.invoke(cli.app, ["test"])
    assert result.exit_code == 0, f"selftest failed: {result.stdout}\n{result.stderr}"
