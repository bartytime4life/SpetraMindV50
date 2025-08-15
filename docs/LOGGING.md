# SpectraMindV50 Logging & Telemetry

- Console + RotatingFile `runtime.log`
- JSONL `events.jsonl` for analytics (`jq`, `pandas.read_json(lines=True)`)
- Per‑run directory: `logs/<RUN_ID>/`
- UTC timestamps; `RUN_START` banner with Git + ENV hash

### JSONL tips
- Tail: `tail -f logs/<id>/events.jsonl | jq .`
- Summarize: `jq -r 'select(.extra.step!=null) | [.ts,.extra.step, .extra.metrics.loss] | @tsv' logs/<id>/events.jsonl`

