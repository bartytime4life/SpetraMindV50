import json
import importlib
import os
import sys
from pathlib import Path
from typer.testing import CliRunner


def _import_cli(monkeypatch, tmp_path: Path):
    """Import the CLI module with deterministic logging for tests."""

    # Write telemetry artefacts to tmp path
    monkeypatch.setenv("SPECTRAMIND_TELEMETRY_LOG_DIR", str(tmp_path))

    # Replace TelemetryManager with a lightweight stub to avoid heavy logging
    class DummyTM:
        def __init__(self, *a, **kw):
            self._diag = None

        def log_event(self, *a, **kw):
            pass

        def log_metric(self, *a, **kw):
            pass

        def attach_diagnostics(self, fn):
            self._diag = fn

        def run_diagnostics(self):
            return self._diag() if self._diag else {}

    for mod in [m for m in list(sys.modules) if m.startswith("spectramind.cli")]:
        del sys.modules[mod]

    import spectramind.cli.cli_telemetry as cli_tel
    monkeypatch.setattr(cli_tel, "TelemetryManager", DummyTM)

    spectramind_cli = importlib.import_module("spectramind.cli.spectramind")
    monkeypatch.setattr(spectramind_cli, "TelemetryManager", DummyTM)
    return spectramind_cli


def test_version_command(tmp_path, monkeypatch):
    cli = _import_cli(monkeypatch, tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["cli_version"] == os.environ.get("SPECTRAMIND_CLI_VERSION")


def test_selftest_fast(tmp_path, monkeypatch):
    cli = _import_cli(monkeypatch, tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["test", "--fast"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["selftest"] == "ok"
