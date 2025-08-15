# bin/ – Executable Scripts for SpectraMind V50

This directory contains CLI wrappers for common project tasks:

- **activate-env.sh** – Activates Python venv and checks CUDA.
- **run-train.sh** – Trains V50 model using Hydra config.
- **run-predict.sh** – Runs inference/prediction.
- **run-diagnose.sh** – Generates diagnostics dashboard.
- **make-submission.sh** – Full pipeline: selftest → train → predict → validate → bundle.
- **selftest.sh** – Runs integrity tests.
- **update-deps.sh** – Updates dependencies via Poetry.
- **launch-dashboard.sh** – Opens latest diagnostics HTML report.
- **ci-run.sh** – Minimal pipeline for CI validation.

All scripts are:
- **Fail-fast** (`set -euo pipefail`)
- **Logging-aware**
- **Hydra-compatible**
- **Reproducibility-focused** (logs time, env, config hash)
