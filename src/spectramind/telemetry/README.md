# SpectraMind V50 Telemetry

The `telemetry` package provides unified telemetry and diagnostics infrastructure for the SpectraMind V50 pipeline.

## Features
- **Rotating File + Console Logging** – Mission-grade logs for CLI and runtime.
- **JSONL Event Stream** – Structured logs for analysis, replay, and dashboards.
- **Metrics Logger** – CSV-based metrics tracking with timestamp and step.
- **Diagnostics Hooks** – Register symbolic/scientific diagnostics for periodic reporting.
- **Hydra Integration** – Options configurable in `telemetry_config.yaml`.
- **Reproducibility** – Logs embed Git commit/branch, ENV hints, and config snapshot hooks.

## Quickstart
```python
from spectramind.telemetry import TelemetryManager

tm = TelemetryManager()
tm.log_event("pipeline_start", {"config_hash": "abc123"})
tm.log_metric("loss", 0.123, step=1)

def custom_diag():
    return {"fft_energy": 42.0}

tm.attach_diagnostics(custom_diag)
tm.run_diagnostics()
```

## Tests

```bash
pytest src/spectramind/telemetry/tests
```
