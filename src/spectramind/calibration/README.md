# SpectraMind V50 – Calibration Package

Mission-grade calibration kill-chain for Ariel-style instruments (FGS1/AIRS), engineered for reproducibility and diagnostics.

## Features

- ADC, nonlinearity, dark, flat-field and cosmic ray removal
- Spectral trace alignment with optional sub-pixel refinement
- Photometry (aperture/optimal) and phase normalisation
- Symbolic calibration (non-negativity, spectral smoothing)
- σ calibration tools: temperature scaling + conformal bin-wise calibration
- Typer CLI with `run`, `batch`, `sigma-calibrate`, `validate` and `selftest`
- Console + rotating file logs and JSONL event stream
- Reproducibility snapshot (Git/ENV capture) and markdown debug log

## Minimal Usage

```bash
python -m spectramind.calibration.calibration_cli selftest
python -m spectramind.calibration.calibration_cli run artifacts/calibration/SELF_AIRS_cube.npy --instrument AIRS
python -m spectramind.calibration.calibration_cli batch artifacts/calibration
```

## Config

See `calibration_config.py` for Hydra-safe dataclasses. `CalibrationConfig.from_yaml()` loads configuration from YAML.

## Outputs

- `artifacts/calibration/{INST}_flux.npy`
- `artifacts/calibration/{INST}_flux_err.npy`
- `artifacts/calibration/{INST}_cube_calibrated.npy`
- `artifacts/calibration/calibration_run.json` or `calibration_batch.json`
- Logs in `logs/`, JSONL event stream in `logs/calibration_events.jsonl`

## License

Apache-2.0
