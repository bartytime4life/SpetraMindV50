import datetime
import json
import os


def write_simple_report(path: str, summary: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><title>SpectraMind V50 Report</title>")
        fh.write("<h1>SpectraMind V50 — Diagnostics Report</h1>")
        fh.write(f"<p>Generated: {datetime.datetime.utcnow().isoformat()}Z</p>")
        fh.write("<pre>")
        fh.write(json.dumps(summary, indent=2))
        fh.write("</pre>")
