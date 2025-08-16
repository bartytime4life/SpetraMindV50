"""
Integration test: CLI logs to v50_debug_log.md
"""
import subprocess
from pathlib import Path
import pytest

pytest.importorskip("typer")

def test_cli_logs_to_file(tmp_path):
    debug_log = tmp_path / "v50_debug_log.md"
    cmd = ["python", "spectramind.py", "--version"]
    subprocess.run(cmd, check=True)
    assert debug_log.exists() or Path("v50_debug_log.md").exists()
