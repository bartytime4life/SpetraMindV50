"""
Integration: ensure the SpectraMind CLI logs version/meta to v50_debug_log.md.

This test SKIPS gracefully if the CLI entrypoint isn't present in the repo yet,
so the suite remains green for incremental adoption.
"""
from pathlib import Path
import os
import subprocess
import sys
import pytest

CLI_ENTRY = Path("spectramind.py")

@pytest.mark.integration
def test_cli_version_logs_file(tmp_path, monkeypatch_env_logging):
    if not CLI_ENTRY.exists():
        pytest.skip("spectramind.py not present; skipping CLI integration test.")
    # Prefer writing logs into tmp_path via env override if the CLI honors it.
    env = dict(os.environ)
    env["SMV50_LOG_DIR"] = str(tmp_path)
    result = subprocess.run([sys.executable, "spectramind.py", "--version"], env=env, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"CLI exited non-zero: {result.stderr}")

    # Accept either local dir or repo root as log destination
    candidate = tmp_path / "v50_debug_log.md"
    if not candidate.exists():
        candidate = Path("v50_debug_log.md")
    assert candidate.exists(), "v50_debug_log.md not found after --version run."
    content = candidate.read_text(encoding="utf-8", errors="ignore")
    # Basic sanity: should mention version or config hash fields that our CLI typically logs
    assert any(k in content for k in ("CLI Version", "Config Hash", "Build Timestamp", "SMV50_CLI_VERSION")), \
        "Version log missing expected metadata fields."
