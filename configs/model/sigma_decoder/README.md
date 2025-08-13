# sigma_decoder (Hydra config group)

This config group defines the **σ (uncertainty) head** for SpectraMind V50 and its **post‑hoc calibration**.

**Files**
- `_group_.yaml` — Hydra defaults (selects `flow`; brings in `calibration`, `monitor`, `export`).
- `flow.yaml` — Heteroscedastic Softplus σ head (default).
- `quantile.yaml` — (q10,q50,q90) head with monotonicity; can derive σ.
- `ensemble.yaml` — Blend flow/quantile σ at inference.
- `calibration.yaml` — Temperature scaling + COREL spectral conformalization.
- `monitor.yaml` — Coverage targets, plots, logging.
- `export.yaml` — Reproducibility artifacts (CSV/JSON/HTML, provenance).

**Usage**
- Default flow head:
  - nothing to change, `defaults` picks `flow`.
- Quantile head:
  - `+sigma_decoder=quantile`
- Ensemble:
  - `+sigma_decoder=ensemble`

**Calibration CLI (examples)**
- Temperature scaling:
  - `python -m spectramind calibrate-temp calibration.temperature.enabled=true`
- COREL conformal:
  - `python -m spectramind calibrate-corel calibration.corel.enabled=true`
