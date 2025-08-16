from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from spectramind.conf_helpers import run_config_audit


def run_config_audit_and_save(
    config_path_or_name: str,
    schema_path: Optional[str],
    overrides: Optional[List[str]],
    out_dir: str = "artifacts/diagnostics",
    filename: str = "config_audit.json",
) -> str:
    """
    Execute config audit and write JSON for dashboard consumption.

    Returns:
        Path to the written JSON file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_json = str(Path(out_dir) / filename)
    run_config_audit(
        config_path_or_name=config_path_or_name,
        schema_path=schema_path,
        overrides=overrides,
        out_json=out_json,
    )
    return out_json
