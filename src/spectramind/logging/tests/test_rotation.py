# SPDX-License-Identifier: MIT
import time
from pathlib import Path

from spectramind.logging import LoggingConfig, get_logger, init_logging


def test_rotation(tmp_path):
    cfg = LoggingConfig(
        log_dir=str(tmp_path),
        file=True,
        console=False,
        jsonl=False,
        file_max_mb=1,
        file_backup_count=2,
    )
    init_logging(cfg)
    log = get_logger("spectramind.rotation")
    msg = "X" * 65536
    for i in range(40):
        log.info("%s_%05d", msg, i)
        time.sleep(0.01)
    base = tmp_path / "spectramind.log"
    assert base.exists()
    rotated_exists = any(
        (tmp_path / f"spectramind.log{suf}").exists() for suf in (".1", ".0", ".01", ".2")
    )
    assert rotated_exists
