import subprocess
import pytest
import importlib.util

torch_missing = importlib.util.find_spec("torch") is None
pytestmark = pytest.mark.skipif(torch_missing, reason="torch not installed")

@pytest.mark.parametrize("command", [
    ["python", "spectramind.py", "--help"],
    ["python", "spectramind.py", "fusion-smoke", "--help"],
    ["python", "spectramind.py", "diagnose", "--help"],
])
def test_cli_help_commands(command):
    """Verify that CLI help commands run without error."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        assert "Usage" in result.stdout or "Options" in result.stdout
    except subprocess.CalledProcessError as e:
        pytest.fail(f"CLI command {command} failed with error: {e.stderr}")
