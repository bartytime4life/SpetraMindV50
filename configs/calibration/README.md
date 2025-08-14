SpectraMind V50 — Calibration Configs

This directory contains Hydra-safe configuration for the calibration kill chain used by SpectraMind V50
before feature extraction and modeling. It encodes reproducible, documented, and CI-validated knobs
for each step:
•Base config (base.yaml) — canonical structure and defaults. All environments compose from this.
•Environment overlays — local.yaml, kaggle.yaml, cluster.yaml.
•Policies (policies.yaml) — IO rules, caching, variance propagation, logging detail, safety guards.
•Steps (steps/*.yaml) — ADC, nonlinearity, dark/flat/bias corrections, trace extraction, photometry,
normalization, temporal alignment, jitter correction, variance propagation.
•Uncertainty calibration — temperature scaling, COREL (graph conformal), conformal fallback.
•Diagnostics (steps/diagnostics.yaml) — CSV/PNG/HTML outputs, quick plots, timing summaries.
•Schema (schema/calibration_config.schema.json) — JSON Schema used in CI to validate structure.

Hydra Composition Examples

Local quick dev:

python -m spectramind calibrate \
  +calib@_global=configs/calibration/base.yaml \
  +calib_env=local \
  +calib_steps=default

Kaggle runtime guardrails:

python -m spectramind calibrate \
  +calib@_global=configs/calibration/base.yaml \
  +calib_env=kaggle \
  steps.diagnostics.enable=true \
  policies.runtime.max_hours=9

Cluster (multi-GPU / fast IO):

python -m spectramind calibrate \
  +calib@_global=configs/calibration/base.yaml \
  +calib_env=cluster \
  io.prefetch_workers=8 io.loader_workers=16

File Map (short)
•base.yaml: master tree (paths, io, policies, steps, logging).
•policies.yaml: centralizes safety, caching, variance rules, overwrite modes.
•local.yaml, kaggle.yaml, cluster.yaml: environment diffs only.
•steps/*.yaml: per-step parameters; imported by base.yaml.
•schema/calibration_config.schema.json: CI/type guard for base.yaml render.

Principles
•Never clip negatives in calibrated frames; carry full variance.
•One source of truth for seeds/workers/logging levels.
•Hydra-safe: no structural changes in env overlays; only values differ.
•Fast-fail diagnostics: misconfiguration detected early via schema + selftest.

