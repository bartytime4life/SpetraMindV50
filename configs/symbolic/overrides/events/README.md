# Event-Driven Overrides

Use these when symbolic rule weights, constraint toggles, or calibration modes should change based on:
- Mission or observation phase (e.g., primary transit, secondary eclipse)
- Instrument state (e.g., FGS lock, AIRS detector heater cycle, jitter excursions)
- Pipeline triggers (e.g., post-calibration, conformalization on/off, anomaly recovery)

Each file adheres to `_schemas/event_override.schema.json`. CI can validate them before merge.
