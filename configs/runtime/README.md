# configs/runtime

This directory defines **runtime environments** for the SpectraMind V50 pipeline.  
Hydra group: `runtime`.

## Files
- **default.yaml** → Base runtime defaults.
- **local.yaml** → Local workstation or laptop dev.
- **kaggle.yaml** → Kaggle competition runtime (9hr GPU limit).
- **hpc.yaml** → HPC cluster jobs with distributed training.
- **docker.yaml** → Containerized runtime execution.
- **ci.yaml** → GitHub Actions or CI/CD jobs.

## Usage
Example CLI calls:
```bash
# Run with local dev runtime
python -m spectramind.cli_core_v50 train runtime=local

# Run with Kaggle runtime
python -m spectramind.cli_submit make-submission runtime=kaggle

# Run with CI runtime
python -m spectramind.selftest runtime=ci
```

All runtime configs are Hydra-safe and interpolate paths, logging, and distributed settings.
They ensure reproducibility, environment portability, and smooth transitions across execution contexts.
