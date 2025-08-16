# configs/paths

This directory defines **all path-related Hydra configs** for SpectraMind V50.

## Overview
- **data.yaml** — raw, processed, cache, and external data dirs
- **outputs.yaml** — checkpoints, predictions, artifacts
- **logs.yaml** — console logs, JSONL streams, debug Markdown, run hash summaries
- **submissions.yaml** — Kaggle submissions, validation runs, ZIP bundles
- **dashboards.yaml** — HTML/JSON diagnostics dashboards, figures, reports
- **config.yaml** — Hydra defaults glue, imports all above

## Key Features
- All paths parameterized with `${oc.env:VAR, default}` for reproducibility
- Compatible with DVC/lakeFS for data versioning
- Integrated with CI/CD + diagnostics dashboard
- Ensures **reproducibility** (no hardcoded paths)

## Usage
```bash
# Example: run training with custom data root
python train_v50.py paths.data.raw_dir=/mnt/new_data/raw
```

This centralization ensures **NASA-grade reproducibility** and easy relocation across servers or Kaggle runtimes.
