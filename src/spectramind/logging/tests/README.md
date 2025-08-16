# SpectraMind V50 — Logging Tests (Mission-Ready)

This suite validates the V50 logging subsystem end-to-end:
- Hydra/structured config presence & minimal schema
- JSONL event stream integrity
- Rotating file handler behavior
- CLI integration (`spectramind.py --version`) with graceful skip if absent
- Symbolic loss JSONL records (rule, loss, metadata)
- Reproducibility env metadata presence (lenient)

## Run
```bash
pytest src/spectramind/logging/tests -v
```

## Markers

* `@pytest.mark.unit` — fast unit checks
* `@pytest.mark.smoke` — minimal end-to-end sanity
* `@pytest.mark.integration` — CLI-level checks (skipped if `spectramind.py` is missing)

## CI

The GitHub Actions workflow `.github/workflows/ci-logging.yml` runs this suite on `push` and `pull_request`,
across Python 3.10–3.11, caches pip, and uploads coverage & pytest JUnit XML.
