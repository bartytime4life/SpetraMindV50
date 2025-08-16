import json
from spectramind.logging.reproducibility import capture_reproducibility_metadata


def test_reproducibility_metadata(tmp_path):
    """Check reproducibility metadata captures ENV + git + config hash."""
    out_file = tmp_path / "meta.json"
    capture_reproducibility_metadata(str(out_file))

    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "env" in data
    assert "git" in data
    assert "config_hash" in data
