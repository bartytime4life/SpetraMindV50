from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spectramind.conf_helpers import capture_environment, capture_environment_detailed


def test_env_capture_basic():
    env = capture_environment()
    assert "python_version" in env
    assert "git" in env


def test_env_capture_detailed():
    env = capture_environment_detailed()
    assert "pip_freeze" in env
