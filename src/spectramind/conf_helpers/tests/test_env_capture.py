import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spectramind.conf_helpers import capture_environment, log_environment


def test_capture_and_log(tmp_path):
    env = capture_environment()
    assert "python_version" in env
    out = tmp_path / "env.json"
    logged = log_environment(out)
    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["python_version"] == logged["python_version"]
