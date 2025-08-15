# Kaggle Auto-Submit Guide (SpectraMind V50)

## Secrets (Settings → Secrets and variables → Actions → Secrets)
- `KAGGLE_USERNAME` – your Kaggle account username
- `KAGGLE_KEY` – your Kaggle API key (JSON value’s key)

## Variables (Settings → Secrets and variables → Actions → Variables)
- `KAGGLE_COMPETITION` – competition slug (e.g., neurips-2025-ariel-data-challenge)
- `AUTO_SUBMIT` – set to "1" to enable auto-submission on pushes to main

## How it works
- CI workflow builds, tests, and attempts to generate `dist/submission.zip`.
- If `AUTO_SUBMIT=1` and secrets exist and branch is main, the auto-submit job:
  - Downloads the CI artifact with `dist/submission.zip`
  - Submits to Kaggle using message containing short SHA and run URL
- Manual submission: run **Actions → Manual Kaggle Submit** and (optionally) provide a message or override the competition slug.

## Expected Bundle

The pipeline expects a `dist/submission.zip` produced by:

```bash
python -m spectramind submit --bundle-out dist/submission.zip --include-html dist/diagnostic_report_v1.html
```

If unavailable, `cli_submit.py` is attempted. Otherwise, submission is skipped gracefully.
