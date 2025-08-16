# conf/hydra

This directory mirrors `configs/hydra` to support parallel Hydra configuration resolution patterns used across different entrypoints and tooling. Keep both trees synchronized.

- `config.yaml`: master Hydra anchors (run/sweep dirs, job meta, defaults).
- `job_logging/default.yaml`: console + rotating file logs + JSONL event stream.
- `launcher/basic.yaml`: local/basic launcher.
- `sweeper/basic.yaml`: basic sweeper for param scans.
- `help.yaml`: CLI help banner/footer for SpectraMind V50.

> Recommendation: Add a CI check to diff `configs/hydra/**` vs `conf/hydra/**` to enforce synchronization.
