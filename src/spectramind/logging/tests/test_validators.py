# SPDX-License-Identifier: MIT
from spectramind.logging import validate_logging_integrity


def test_validate_logging_integrity(tmp_path):
    out = validate_logging_integrity(tmp_dir=str(tmp_path), level="INFO", check_rotation=True)
    assert out["status"] == "ok"
    assert any(c["check"] == "file_rotation_happened" and c["ok"] for c in out["checks"])
