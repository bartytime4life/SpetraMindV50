# SpectraMind V50 Logging Directory

This directory stores **all runtime logs** produced by the SpectraMind V50 pipeline, CLI, CI jobs, and diagnostics tools. It complements Hydra’s job logging group at `configs/hydra/job_logging/*` and is designed for **reproducibility**, **post-hoc analysis**, and **forensic debugging** across machines and CI.

---

## What gets written here

1. Rotating text logs (human-friendly)

   * File: `logs/v50_pipeline.log`
   * Source: Python `logging` via `RotatingFileHandler`
   * Purpose: Readable, timestamped trace of the session (INFO+ by default on console; DEBUG to file)

2. JSONL event stream (machine-friendly)

   * File: `logs/v50_event_stream.jsonl`
   * Source: Python `logging` using a JSON formatter
   * Purpose: Structured logs for programmatic analysis (pandas, jq, dashboards)
   * One event per line; safe to tail, stream, compress, or ship

3. Optional MLflow/W\&B mirror handlers

   * Enabled via Hydra overrides:

     * `+hydra/job_logging=mlflow`
     * `+hydra/job_logging=wandb`
   * Purpose: Emit mirrored signals to experiment trackers (does not replace local files)

---

## File lifecycle & rotation policy

* `v50_pipeline.log` rotates automatically when it reaches ~50 MB (retains 10 backups):

  * `v50_pipeline.log` (active)
  * `v50_pipeline.log.1` … `v50_pipeline.log.10` (oldest)
* `v50_event_stream.jsonl` does **not** rotate by default (append-only) to keep line-addressable event history.

  * You may archive/compress periodically in CI (e.g., nightly): `gzip logs/v50_event_stream.jsonl-YYYYMMDD.gz`

> Rationale: human log rotates for readability; JSONL is preferred for analytics and can be chunked by date in CI.

---

## JSONL schema

Each line is a JSON object with the following keys:

* `time` (str, ISO-like): wall-clock timestamp, e.g., `"2025-08-15T21:42:33"`
* `level` (str): Python logging level, e.g., `"INFO"`, `"DEBUG"`, `"WARNING"`, `"ERROR"`
* `name` (str): logger name (`spectramind.*`, `hydra`, etc.)
* `process` (int): OS process id
* `message` (str): the formatted log message

Example line:
{"time":"2025-08-15T21:42:33", "level":"INFO", "name":"spectramind.cli", "process":12345, "message":"Started run: cli=diagnose dashboard version=0.50.0 config_hash=abc123"}

> You may enrich messages with key=value tokens (e.g., `planet=ARIEL_0113 phase=train gll=1.234`) to simplify downstream parsing.

---

## How to consume logs

### Quick human reads

* View last 100 lines:

  * `tail -n 100 logs/v50_pipeline.log`
  * `tail -n 100 logs/v50_event_stream.jsonl`
* Follow in real time:

  * `tail -f logs/v50_pipeline.log`
  * `tail -f logs/v50_event_stream.jsonl`

### Filter with jq (JSONL)

* All ERRORs:

  * `jq -rc 'select(.level=="ERROR")' logs/v50_event_stream.jsonl`
* Only `spectramind.diagnose` namespace:

  * `jq -rc 'select(.name|startswith("spectramind.diagnose"))' logs/v50_event_stream.jsonl`
* Extract time + message:

  * `jq -rc '[.time,.message] | @tsv' logs/v50_event_stream.jsonl`

### Load with pandas (Python)

```
import pandas as pd

df = pd.read_json("logs/v50_event_stream.jsonl", lines=True)
df_time = pd.to_datetime(df["time"])
err = df[df["level"].isin(["ERROR","CRITICAL"])]
print(err[["time","name","message"]].tail(20))
```

---

## Integration with Hydra job logging

The default config is `configs/hydra/job_logging/spectramind_v50.yaml`. It defines:

* Console handler (INFO to stdout)
* Rotating file handler (`logs/v50_pipeline.log`, DEBUG level)
* JSONL file handler (`logs/v50_event_stream.jsonl`, DEBUG level)

You can extend handlers (e.g., MLflow/W&B) using:

* `+hydra/job_logging=mlflow`
* `+hydra/job_logging=wandb`

> Handlers are additive—overrides append to `root.handlers`.

---

## Environment & paths

* Files are relative to the repository working directory by default.
* Ensure the `logs/` directory exists before first write (created on repo checkout). If you run from a different CWD, either:

  * Run CLI from repo root, or
  * Configure absolute paths by overriding handler filenames in Hydra:

    * `hydra.job_logging.spectramind_v50.handlers.file.filename=/abs/path/logs/v50_pipeline.log`
    * `hydra.job_logging.spectramind_v50.handlers.jsonl.filename=/abs/path/logs/v50_event_stream.jsonl`

---

## Log level policy

* **Root logger**: `DEBUG` (so all handlers can capture detailed info to files)
* **Console**: `INFO` (keep output readable for humans)
* **Files (text & JSONL)**: `DEBUG` (full fidelity for audits and diagnostics)
* **Hydra logger**: `INFO` to reduce verbosity on framework internals

Override at runtime (Hydra):

* `hydra.verbose=true` or `hydra.verbose=false`
* Per-logger override example:

  * `hydra.job_logging.spectramind_v50.loggers.spectramind.level=INFO`

---

## CI & archival recommendations

* Persist artifacts from `logs/` as GitHub Actions workflow artifacts on failures and on main branch nightly.
* Compress large JSONL weekly or split by date (e.g., rotate to `v50_event_stream-YYYY-MM-DD.jsonl` via cron or CI).
* For dashboards, ingest JSONL into your analytics step to render heatmaps, timelines, or error trend charts.

---

## PII / secrets hygiene

* Avoid logging secrets, API keys, tokens, or raw data that could be sensitive.
* If needed, implement redaction in your logging calls before emitting messages:

  * E.g., transform `api_key=sk-123...` → `api_key=****`
* Keep JSONL public-safe by default; if you must include sensitive context, store in a separate, access-controlled path.

---

## Troubleshooting

* Nothing is written?

  * Confirm you launched from repo root, or adjust handler filenames to absolute paths.
  * Confirm process user has write permissions for `logs/`.
  * Verify your CLI/module uses the `spectramind` logger or `logging.getLogger(__name__)`.
* Log spam?

  * Increase console level: set console handler `level: WARNING` via Hydra override.
  * Suppress noisy third-party loggers by adding entries under `loggers:` with higher thresholds.

---

## Provenance & reproducibility

* Log banner lines should include: CLI version, config hash, git commit, env snapshot, and timestamp.
* The JSONL file is the canonical source for machine parsing in the diagnostics dashboard, CI, and audits.

---

## Related configs

* `configs/hydra/job_logging/default.yaml`
* `configs/hydra/job_logging/spectramind_v50.yaml`
* `configs/hydra/job_logging/mlflow.yaml`
* `configs/hydra/job_logging/wandb.yaml`

Keep these in sync if you change handler filenames, formats, or retention policies.
