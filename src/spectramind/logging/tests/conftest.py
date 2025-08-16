"""
Pytest configuration for logging tests.
Adds fixtures for temp log directories and mock CLI runs.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_log_dir():
    """Fixture to create a temporary log directory for tests."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)
