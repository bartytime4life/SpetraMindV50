# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for SpectraMind CLI tests.

These fixtures provide a Typer `CliRunner`, isolate the working directory to a
temporary path and seed minimal repository files expected by some commands.
Environment variables are also adjusted so telemetry and logging output stay
within the temporary directory used for each test.
"""
import json
import os
import sys
from pathlib import Path
from typing import Callable

import pytest
from typer.testing import CliRunner

# Ensure the project's `src` directory is importable when tests are executed
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def runner() -> CliRunner:
    """Return a Typer CliRunner instance for invoking commands."""
    return CliRunner()


@pytest.fixture
def monkeypatch_cwd_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Change the working directory to a temporary path for the test."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def env_test_mode(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Set common environment variables so the CLI behaves in test mode."""
    monkeypatch.setenv("SPECTRAMIND_TEST", "1")
    monkeypatch.setenv("SPECTRAMIND_DRYRUN_DEFAULT", "1")
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    # Keep telemetry and log artifacts local to the current working directory
    monkeypatch.setenv("SPECTRAMIND_LOG_DIR", ".")
    return True


@pytest.fixture
def repo_tmp_files(monkeypatch_cwd_tmp: Path) -> bool:
    """Seed minimal files some commands expect to exist."""
    Path("v50_debug_log.md").write_text(
        "# SpectraMind V50 Debug Log\n"
        "2025-08-16T12:00:00Z | CLI=1.0.0 | config_hash=deadbeef | cmd='spectramind --version'\n"
    )
    Path("run_hash_summary_v50.json").write_text(
        json.dumps(
            {
                "config_hash": "deadbeef",
                "build_timestamp": "2025-08-16T12:00:00Z",
                "cli_version": "1.0.0",
            },
            indent=2,
        )
    )
    return True


@pytest.fixture
def json_tmp_path_factory(tmp_path: Path) -> Callable[[str], Path]:
    """Return a factory that builds output paths within `tmp_path`."""
    def factory(name: str) -> Path:
        out = tmp_path / name
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    return factory
