import os
from typing import Dict, Any

from .paths import get_default_paths, ensure_dir
from .logging_utils import get_logger
from .io_utils import save_json, load_json


def utils_selftest() -> Dict[str, Any]:
    """Fast integrity test for the utils subsystem: logging, paths, I/O."""
    out: Dict[str, Any] = {"ok": True, "steps": []}
    log = get_logger("utils.selftest")
    paths = get_default_paths()

    # 1) Paths and directories
    ensure_dir(paths["artifacts_dir"])
    out["steps"].append({"paths_ok": True})

    # 2) Write/read JSON
    test_obj = {"hello": "world", "k": 42}
    test_json = os.path.join(paths["artifacts_dir"], "utils_selftest.json")
    save_json(test_obj, test_json)
    roundtrip = load_json(test_json)
    out["steps"].append({"json_roundtrip_ok": (roundtrip == test_obj)})

    # 3) Log an event
    log.info("utils_selftest event")
    out["steps"].append({"logged_ok": True})

    # Aggregate
    out["ok"] = all(step.get(list(step.keys())[0], True) for step in out["steps"])
    return out
