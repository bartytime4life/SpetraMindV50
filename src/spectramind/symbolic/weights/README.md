# SpectraMind V50 — Symbolic Weights Subsystem

This package provides:

- Declarative YAML weight sets for symbolic constraints
- Deterministic composition and validation
- Metric-driven optimization with bounded, conservative updates
- Mission-grade logging (console + rotating file) and JSONL event stream
- Typer CLI (`python -m spectramind.symbolic.weights.cli`)

## Quickstart

Compose defaults:

```bash
python -m spectramind.symbolic.weights.cli compose
```

Compose with composite profile:

```bash
python -m spectramind.symbolic.weights.cli compose --profile-name hot_jupiter --json-out weights_hot_jupiter.json
```

Optimize from metrics:

```bash
python -m spectramind.symbolic.weights.cli optimize weights_hot_jupiter.json metrics.json --out-json weights_hot_jupiter_optimized.json
```

Logs: `logs/symbolic_weights.log`, events: `logs/symbolic_weights.events.jsonl`.
