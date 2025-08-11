# Operator Quick Reference

A cheat sheet for operating SpectraMind locally and on Kaggle.

## Local mode
- Install dependencies and activate the environment.
- Run training:
  ```bash
  python -m spectramind train
  ```
- Perform calibration and diagnostics:
  ```bash
  python -m spectramind calibrate-temp
  python -m spectramind diagnose
  ```
- Create a submission bundle:
  ```bash
  python -m spectramind submit bundle
  ```

## Kaggle mode
- Use the Kaggle runtime and data paths (e.g. `/kaggle/input` and `/kaggle/working`).
- Select the `data=kaggle` config override when invoking the CLI.
- Typical prediction entry point:
  ```bash
  python -m spectramind predict --out-csv /kaggle/working/submission.csv
  ```
- Keep the total runtime under Kaggle's 9‑hour limit.
