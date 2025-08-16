from pathlib import Path

from spectramind.telemetry import TelemetryManager


def test_event_and_metric_logging(tmp_path: Path) -> None:
    tm = TelemetryManager(log_dir=str(tmp_path), enable_jsonl=True)
    tm.log_event("test_event", {"msg": "hello"})
    tm.log_metric("accuracy", 0.95, step=1)

    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    metrics_file = tmp_path / "metrics.csv"
    assert metrics_file.exists()
    content = metrics_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) >= 2
    assert content[0].split(",") == ["timestamp", "step", "metric_name", "value"]


def test_diagnostics_hooks(tmp_path: Path) -> None:
    tm = TelemetryManager(log_dir=str(tmp_path))
    tm.attach_diagnostics(lambda: {"check": "ok", "value": 7})
    results = tm.run_diagnostics()
    assert "check" in results and results["check"] == "ok"
    assert "value" in results and results["value"] == 7
