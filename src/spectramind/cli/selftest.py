import json
import sys
from pathlib import Path
from typing import Dict, List

from .common import (
    PROJECT_ROOT,
    SRC_DIR,
    PKG_DIR,
    CLI_DIR,
    LOG_DIR,
    REPORTS_DIR,
    DIAG_DIR,
    logger,
    capture_env_snapshot,
    write_jsonl_event,
    md_table,
    command_session,
)

REQUIRED_PATHS = [
    SRC_DIR / "spectramind",
    PKG_DIR / "__init__.py",
]

RECOMMENDED_FILES = [
    PROJECT_ROOT / "config_v50.yaml",
    PROJECT_ROOT / "README.md",
]

OPTIONAL_SCRIPTS = [
    SRC_DIR / "spectramind" / "train_v50.py",
    SRC_DIR / "spectramind" / "predict_v50.py",
    SRC_DIR / "spectramind" / "generate_html_report.py",
    SRC_DIR / "spectramind" / "generate_diagnostic_summary.py",
]

def run_selftest(mode: str = "fast") -> Dict[str, str]:
    status: Dict[str, str] = {}
    # Basic structure checks
    for p in REQUIRED_PATHS:
        status[str(p.relative_to(PROJECT_ROOT))] = "ok" if p.exists() else "missing"
    # Recommended
    for p in RECOMMENDED_FILES:
        status[str(p.relative_to(PROJECT_ROOT))] = "ok" if p.exists() else "missing"
    # Optional scripts
    for p in OPTIONAL_SCRIPTS:
        status[str(p.relative_to(PROJECT_ROOT))] = "ok" if p.exists() else "missing"
    # Import tests
    if mode != "deep":
        try:
            import typer  # noqa: F401
            status["python.typer"] = "ok"
        except Exception as e:
            status["python.typer"] = f"fail:{e}"
    else:
        modules = [
            "spectramind.train_v50",
            "spectramind.predict_v50",
            "spectramind.generate_html_report",
            "spectramind.generate_diagnostic_summary",
        ]
        for m in modules:
            try:
                __import__(m)
                status[f"import.{m}"] = "ok"
            except Exception as e:
                status[f"import.{m}"] = f"fail:{e}"
    return status


def write_reports(status: Dict[str, str]) -> None:
    # JSON
    rep_json = REPORTS_DIR / "selftest_report.json"
    rep_json.write_text(json.dumps(status, indent=2), encoding="utf-8")
    # Markdown table
    rows = [["Item", "Status"]] + [[k, v] for k, v in sorted(status.items())]
    md = "# SpectraMind V50 — Selftest Report\n\n" + md_table(rows) + "\n"
    (REPORTS_DIR / "selftest_report.md").write_text(md, encoding="utf-8")
    logger.info("Wrote: %s", rep_json)
    logger.info("Wrote: %s", REPORTS_DIR / "selftest_report.md")


def cli(mode: str = "fast") -> int:
    with command_session("selftest", [f"--mode={mode}"]):
        status = run_selftest(mode=mode)
        write_reports(status)
        envdir = REPORTS_DIR / "env"
        snaps = capture_env_snapshot(envdir)
        write_jsonl_event("selftest_done", {"mode": mode, "status": status, "env": snaps})
        ok = all(v == "ok" for v in status.values() if not v.startswith("fail"))
        return 0 if ok else 1


if __name__ == "__main__":
    m = "fast"
    if len(sys.argv) > 1 and sys.argv[1].startswith("--mode="):
        m = sys.argv[1].split("=", 1)[1]
    sys.exit(cli(m))
