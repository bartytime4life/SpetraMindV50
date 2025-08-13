
⸻

SpectraMind V50 — Debug & Audit Log

Immutable operator log — append-only. Every CLI invocation, config, and key result is recorded here for full reproducibility ￼.

⸻

2025-08-12T03:22:45Z — spectramind selftest
	•	Git SHA: d8e4c1f
	•	Config hash: cb61a1c
	•	Host: ws-rtx508-dev01
	•	CUDA / Driver: 12.1 / 545.29
	•	Seed: 133742
	•	Hydra overrides: phase=selftest
	•	Result: ✅ All files present, CLI commands registered, configs resolved, dry-run passed.
	•	Artifacts: outputs/selftest/report.md, outputs/selftest/report.json

⸻

2025-08-12T03:34:10Z — spectramind calibrate --dry-run
	•	Git SHA: d8e4c1f
	•	Config hash: 11f8d24
	•	Hydra overrides: data=kaggle
	•	Calibration plan:
	1.	ADC reversal
	2.	Bad-pixel masking
	3.	Non-linearity correction
	4.	Dark subtraction
	5.	Flat-fielding
	6.	Trace extraction ￼
	•	Result: ✅ Dry-run complete — 0 planets processed (planning only).
	•	Artifacts: logs/calibration_plan.json

⸻

2025-08-12T04:05:18Z — spectramind train phase=mae
	•	Git SHA: 34ad8b9
	•	Config hash: f391b22
	•	Hydra overrides: phase=mae data=local trainer.max_epochs=1
	•	FGS1 encoder: Mamba(d_state=64, bidir=True)
	•	AIRS encoder: GAT(num_layers=3, edge_feats=True) ￼
	•	Symbolic loss pack: smoothness, non-negativity, molecular coherence enabled.
	•	Result: ✅ 1 epoch complete — val_GLL=-4123.5.
	•	Artifacts: outputs/train/mae/epoch1.ckpt, outputs/train/mae/metrics.json

⸻

2025-08-12T05:18:49Z — spectramind calibrate-temp
	•	Git SHA: 34ad8b9
	•	Config hash: f391b22
	•	Hydra overrides: calibration.method=temp_scale
	•	Result: ✅ Temperature scaling applied — val_GLL improved from -4123.5 → -4010.8.
	•	Artifacts: outputs/calibration/temp_scaling_params.json, outputs/calibration/plots/

⸻

2025-08-12T05:44:27Z — spectramind diagnose dashboard --html
	•	Git SHA: 34ad8b9
	•	Config hash: f391b22
	•	Hydra overrides: dashboard.version=v1
	•	Diagnostics: SHAP×symbolic overlays, UMAP, t-SNE, FFT, smoothness maps ￼.
	•	Result: ✅ HTML report generated.
	•	Artifacts: outputs/diagnostics/dashboard_v1.html

⸻

2025-08-12T06:15:01Z — spectramind submit bundle
	•	Git SHA: 34ad8b9
	•	Config hash: f391b22
	•	Hydra overrides: data=kaggle
	•	Result: ✅ Submission bundle validated (μ/σ × 283 bins) and zipped.
	•	Artifacts: outputs/submission/submission.csv, outputs/submission/submission_bundle.zip, run_hash_summary_v50.json

⸻


---
### 2025-08-13T17:40:44Z — sigma_decoder self-test
- Result: **PASS**
- Files checked: 7
  - `_group_.yaml`: `f90d895734f23208cd6154675ca43b7dbdbecafdf8658ddeeb79b9f4776cee13`
  - `flow.yaml`: `e65e34cdb197f0ba1f2e97c17535f5a428640f33342e7b9c33f00d647c916cb1`
  - `quantile.yaml`: `879f6bf63d948e52ea114ad04c9d6c4cd58ac391dd9d308a5064e9e79f292dd6`
  - `ensemble.yaml`: `9b5247a91d8f3b3961301b7b0461996bb8119c09ba096750ff71f48b87179c87`
  - `calibration.yaml`: `df3961801924674da2c570915b8e0ccb1dee09b045d2381737da41d64a325580`
  - `monitor.yaml`: `1e16d65ccb8137ca4d94de4d0560374095256d0c915895ed57ac1eacb0bb5164`
  - `export.yaml`: `10134b4b9b9f42c16e5007423a49fb0ed2614e5638f9546bbb08d1182cb0bdcf`
- Notes: configs present; keys validated; ensemble weights sane; coverage targets sane.
