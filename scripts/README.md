# scripts/ — SpectraMind V50 Utility Scripts

This directory contains CLI-friendly wrappers and maintenance utilities designed to make the SpectraMind V50 workflow fast, reproducible, and push-ready. Each script uses strict Bash settings (`set -Eeuo pipefail`), writes timestamped logs under `logs/`, and integrates with the repository’s hashing and diagnostics systems.

## Contents
- `train_v50.sh` — Run training with Hydra configs, full logging, and hashing.
- `predict_v50.sh` — Run inference with a specified checkpoint, write outputs, and optionally bundle for Kaggle.
- `calibrate_v50.sh` — Execute the full calibration pipeline (photometry, σ calibration incl. COREL).
- `run_selftest.sh` — Run repository self-tests (fast or deep) and optional log cleanup.
- `launch_dashboard.sh` — Open a generated diagnostics HTML in your browser.
- `bundle_submission.sh` — Package μ/σ predictions into a Kaggle-ready ZIP with manifest.
- `clean_logs.sh` — Deduplicate `v50_debug_log.md`, trim old logs, and rotate console logs.
- `hash_config.sh` — Compute config SHA-256 + record run metadata to `run_hash_summary_v50.json`.
- `ci_check.sh` — Local CI mimic: unit tests, selftest, optional lint/format checks.
- `docker_build.sh` — Build a pinned Docker image for reproducible runs.
- `export_env.sh` — Export Poetry/Conda/pip environment snapshots and system info.
- `analyze_log.sh` — Parse `v50_debug_log.md` into CSV/Markdown summaries; optional dedupe.
- `diagnose_dashboard.sh` — Generate the unified diagnostics dashboard HTML.
- `generate_dummy_data.sh` — Produce synthetic AIRS/FGS1 test data for pipeline checks.
- `profile_diagnose.sh` — Inspect symbolic profile usage across all planets.
- `symbolic_rank.sh` — Rank symbolic rule violations and export results.
- `tsne_latents.sh` — Produce interactive t-SNE latent visualizations.
- `umap_latents.sh` — Produce interactive UMAP latent visualizations.
- `check_cli_map.sh` — Export the command↔file mapping and optionally open the page.
- `corel_train.sh` — Train the COREL calibrator with full logging.
- `push_ready_check.sh` — Ensure the repo is clean and tests pass prior to pushing.
- `one_click_make_submission.sh` — E2E: selftest → predict → bundle (→ open HTML).

## Conventions
- All scripts assume the root CLI is invokable via:
  - `python -m src.spectramind.spectramind <subcommands>` (default), or
  - `uv run python -m ...`, or `poetry run python -m ...` if available.
- Logs live inside `logs/<tool>_<YYYYMMDD>_<HHMMSS>[_tag]/`.
- Hashes are appended to `run_hash_summary_v50.json`.
- HTML dashboard helpers will try to open with `xdg-open`/`open`/`start`.

## Quickstarts
- Train:
  ```bash
  bash scripts/train_v50.sh --config configs/model/config_v50.yaml --tag baseline
  ```
- Predict & bundle:
  ```bash
  bash scripts/predict_v50.sh --ckpt ckpts/best.pt --bundle --open-html
  ```
- One-click submission:
  ```bash
  bash scripts/one_click_make_submission.sh --ckpt ckpts/best.pt --open-html
  ```
- Diagnostics dashboard:
  ```bash
  bash scripts/diagnose_dashboard.sh --open
  ```

Make scripts executable:
```bash
chmod +x scripts/*.sh
```

All scripts are production-ready, Hydra-safe, and integrate into the SpectraMind V50 reproducibility stack.
