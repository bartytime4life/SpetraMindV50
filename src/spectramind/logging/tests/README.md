# SpectraMind V50 Logging Tests

This directory provides **mission-ready test coverage** for the logging subsystem.

## Features Tested
- Hydra logging config parsing
- JSONL event stream validity
- Rotating file log behavior
- CLI integration (`spectramind --version`)
- JSONL formatting
- Symbolic loss logging

## Usage
Run all tests:

```bash
pytest src/spectramind/logging/tests -v
```

## CI/CD

These tests run in GitHub Actions to guarantee reproducibility and logging integrity.
