# SpectraMind V50 — Reporting Schemas

This package defines **mission-grade Pydantic schemas** for the SpectraMind V50 reporting layer. It provides:

* Strongly-typed, reproducible data models for diagnostics, symbolic violations, calibration, HTML assets, and full report packages.
* A **schema registry** and an **export utility** to generate JSON Schemas for downstream tools (UI, CI, or validation).
* A lightweight **self-test** to validate model instantiation and schema export.

## Contents

* `base.py` – Common `BaseSchema` with reproducibility metadata and deterministic serialization.
* `types.py` – Reusable constrained type aliases (e.g., `HashStr`, `PlanetId`, `IsoDatetime`).
* `cli_log.py` – CLI log entry and batch summary schemas.
* `diagnostic_summary.py` – Global and per-bin diagnostic schemas.
* `symbolic_violation.py` – Symbolic rule hit and per-planet violation summary schemas.
* `calibration.py` – Coverage and per-bin calibration quality.
* `html_report.py` – HTML asset and report manifest schemas with file verification helper.
* `report_manifest.py` – The **ReportPackageManifest** that assembles everything for publication.
* `schema_registry.py` – Global registry with JSON Schema exporter.
* `export.py` – Command to export all JSON Schemas.
* `selftest.py` – Minimal runtime check.

## Quick Start

Export JSON Schemas to `.artifacts/schemas`:

```bash
python -m spectramind.reporting.schema.export --out .artifacts/schemas
```

Run quick self-test:

```bash
python -m spectramind.reporting.schema.selftest
```

## Notes

* Models are Pydantic v2-compatible and serialize deterministically for hashing and CI diffs.
* `BaseSchema` injects `schema_version`, `created_at`, `git_commit`, and optional `run_hash`.
* The registry enables discovery without hard-coding model lists in exporter/validators.
