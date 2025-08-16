"""
Pytest configuration & fixtures for SpectraMind V50 logging tests.

Fixtures:
- temp_log_dir: ephemeral directory for log outputs
- monkeypatch_env_logging: sets env vars expected by logging to test reproducibility fields
"""
import os
import shutil
import tempfile
from pathlib import Path
import pytest

@pytest.fixture
def temp_log_dir():
    d = Path(tempfile.mkdtemp(prefix="smv50_logs_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

@pytest.fixture
def monkeypatch_env_logging(monkeypatch, tmp_path):
    # Common env knobs SpectraMind uses for logging; safe if unused.
    monkeypatch.setenv("SMV50_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("SMV50_RUN_ID", "test-run-0001")
    monkeypatch.setenv("SMV50_CONFIG_HASH", "deadbeefcafebabe")
    monkeypatch.setenv("SMV50_CLI_VERSION", "v50.dev-test")
    return tmp_path
