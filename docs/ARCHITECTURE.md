⸻

SpectraMind V50 — Mission Architecture & Engineering Doctrine

ESA Ariel × NeurIPS 2025 Challenge – Physics-Informed, Neuro-Symbolic AI for Exoplanet Spectroscopy

⸻

Executive Summary

SpectraMind V50 is a mission-grade AI system designed to win the NeurIPS 2025 Ariel Data Challenge while setting new standards for scientific fidelity, reproducibility, and explainability.

Its architecture fuses:
	•	Deep temporal and spectral encoders tailored for the Ariel telescope’s FGS1 and AIRS instruments.
	•	Physics-informed symbolic constraints that encode universal atmospheric and instrument truths.
	•	Post-hoc uncertainty calibration (Temperature Scaling + COREL conformal prediction) to maximize leaderboard score under the Gaussian Log-Likelihood metric.
	•	CLI-first, Hydra-configured MLOps stack to guarantee reproducibility, speed, and operational control.

The system is engineered to:
	•	Complete inference for ~1,100 planets in ≤ 9 hours on Kaggle GPUs.
	•	Produce μ/σ spectra that are physically plausible, uncertainty-calibrated, and robust to out-of-distribution (OOD) atmospheres.
	•	Provide operator-first explainability via SHAP × symbolic overlays, UMAP/t-SNE visualizations, FFT smoothness maps, and interactive HTML dashboards.

⸻

Table of Contents
	1.	Scientific & Competitive Landscape
	2.	End-to-End System Dataflow
	3.	Data Contracts & Schemas
	4.	Calibration Kill Chain
	5.	Modeling System
	6.	Objective: Likelihood + Symbolic Physics
	7.	Uncertainty Calibration
	8.	Training Curriculum & Schedules
	9.	CLI-First MLOps & Config Discipline
	10.	Diagnostics & Dashboard
	11.	CI/CD Doctrine
	12.	Reproducibility Stack
	13.	Failure Modes & Runbooks
	14.	Symbolic Logic Engine
	15.	Exact Config Knobs
	16.	Tests
	17.	ADR & Change Control
	18.	Operational Quick Start
	19.	Engineering Deep Dive

⸻

1) Scientific & Competitive Landscape

Observational Physics Context
	•	Transmission spectroscopy detects wavelength-dependent absorption in an exoplanet’s atmosphere during transit.
	•	The atmospheric scale height
H = \frac{kT}{\mu g}
governs feature depths — hotter/lower-g planets exhibit stronger signatures.
	•	FGS1 (0.60–0.80 μm broadband) dominates the leaderboard weight (~58× a single AIRS bin) and serves as the global transit reference.
	•	AIRS-CH0 (1.95–3.90 μm) contains molecular fingerprints (H₂O, CH₄, CO₂) over 283 bins.

Challenge Constraints
	•	Metric: Gaussian Log-Likelihood (GLL) — rewards accurate μ and honest σ.
	•	Strict Kaggle runtime budget: ≤ 9 h for ~1,100 planets.
	•	Test set is OOD — requiring physical priors to generalize beyond training distributions.

⸻

2) End-to-End System (Level-1 Dataflow)

flowchart LR
  subgraph RAW[Raw Inputs]
    A[FGS1 cubes]
    B[AIRS-CH0 cubes]
  end

  subgraph CAL[Calibration Kill Chain]
    C1[ADC reversal]-->C2[Bad pixel mask]-->C3[Non-linearity]
    C3-->C4[Dark subtraction]-->C5[Flat-field]-->C6[Trace extraction]
  end

  subgraph FEAT[Feature Cache]
    F1[FGS1 white-light]
    F2[AIRS per-bin (283)]
    F3[Jitter/centroid]
  end

  subgraph MODEL[Encoders→Fusion→Decoders]
    M1[FGS1 Mamba SSM]
    M2[AIRS GAT (283 nodes)]
    MF[Fusion Layer]
    D1[Multi-Scale μ Decoder]
    D2[Flow/Quantile σ Head]
  end

  subgraph SYM[Symbolic Physics Engine]
    S1[Smoothness]
    S2[Non-negativity]
    S3[Molecule coherence]
    S4[Seam continuity]
    S5[CH4:H2O:CO2 envelopes]
    S6[Quantile monotonicity]
  end

  subgraph CAL2[Uncertainty Calibration]
    T1[Temperature scaling]
    T2[COREL conformal prediction]
  end

  RAW-->CAL-->FEAT
  FEAT-->M1 & FEAT-->M2
  M1-->MF & M2-->MF
  MF-->D1 & MF-->D2
  D1 & D2-->SYM
  SYM-->CAL2
  CAL2-->OUT[submission.csv]


⸻

3) Data Contracts & Schemas

Invariant: Never clip negatives in calibrated frames; always propagate variance; log correction magnitudes.

Directory Layout

/data
  raw/{fgs1,airs_ch0}/...
  calibrated/{fgs1,airs}/...
  features/{fgs1_white,airs_bins}/...
  splits/groupkfold.json

Calibrated Tensor Format

File	Keys	Shape	Notes
fgs1_{planet}.npz	frames, variance, mask	(T,H,W)	float32/bool
airs_{planet}.npz	frames, variance, mask, trace_meta	(T,H,W)	Includes slit, seam, dispersion poly


⸻

4) Calibration Kill Chain

Why it matters: Errors here are unrecoverable downstream. Each step is DVC-tracked and validated.

graph TD
  A[ADC Reversal] --> B[Bad Pixel Mask/Interp]
  B --> C[Non-Linearity Correction]
  C --> D[Dark Subtraction]
  D --> E[Flat-Fielding]
  E --> F[Trace Extraction / Photometry]


⸻

5) Modeling System

Encoders
	•	FGS1 Mamba SSM — optimized for long temporal sequences; bidirectional option; TorchScript safe.
	•	AIRS GAT — graph with λ-adjacency, molecule, and seam edges; attention export for explainability.

Fusion — concat+MLP, cross-attend, or gated.

Decoders
	•	Multi-Scale μ — coarse→fine with skip fusion.
	•	σ Head — Flow or quantile; compatible with temp scaling + COREL.

⸻

6) Objective: Likelihood + Symbolic Physics

GLL Loss
\mathcal{L}_{gll} = \frac{1}{2} \sum_i \left[ \log(2\pi\sigma_i^2) + \frac{(y_i - \mu_i)^2}{\sigma_i^2} \right]

Symbolic Loss Pack
	•	Smoothness (2nd derivative)
	•	Non-negativity
	•	Molecular coherence
	•	Seam continuity
	•	Chemistry envelopes
	•	Quantile monotonicity

⸻

7) Uncertainty Calibration
	1.	Temperature Scaling — fit τ to validation GLL.
	2.	COREL Conformal Prediction — per-bin coverage targeting, molecule-region coverage plots.

⸻

8) Training Curriculum & Schedules
	1.	MAE pretrain — physics-aware masking.
	2.	Contrastive — InfoNCE on in/out-of-transit segments.
	3.	Supervised — GLL + λ·Symbolic.
	4.	Calibration — Temp scaling → COREL.

⸻

9) CLI-First MLOps & Config Discipline
	•	Typer CLI — one entry point: spectramind.py
	•	Hydra Configs — hierarchical, CLI-overridable.
	•	Logging — human (v50_debug_log.md) + machine (events.jsonl) logs.

⸻

10) Diagnostics & Dashboard

Interactive HTML includes:
	•	UMAP/t-SNE latent projections.
	•	SHAP × symbolic overlays.
	•	FFT smoothness maps.
	•	COREL coverage plots.
	•	CLI log summary.

⸻

11) CI/CD Doctrine

sequenceDiagram
  participant Dev
  participant CI as GitHub Actions
  Dev->>CI: Push PR
  CI->>CI: Build Env
  CI->>CI: Selftest
  CI->>CI: Unit Tests
  CI->>CI: Smoke E2E
  CI->>CI: Pipeline Consistency Check
  CI->>Dev: Upload Artifacts


⸻

12) Reproducibility Stack
	•	Git SHA + config hash.
	•	Poetry/Docker env pinning.
	•	DVC/lakeFS for datasets/models.
	•	Immutable run manifest.

⸻

13) Failure Modes & Runbooks
	•	σ underestimation — run calibrate-temp → calibrate-corel.
	•	Seam discontinuity — adjust λ_seam; verify AIRS trace.
	•	FGS1 drift — increase Mamba capacity; add limb-darkening features.
	•	Runtime > budget — use overrides/kaggle_fast.yaml.

⸻

14) Symbolic Logic Engine
	•	Evaluates μ/σ/quantiles against symbolic rules.
	•	Outputs per-rule violation logs.
	•	Generates influence maps: ∂L_sym / ∂μ.

⸻

15) Exact Config Knobs

Section	Key Parameters
model.fgs1_mamba	d_model=256, n_layers=8, bidir=true, dropout=0.1
model.airs_gnn	d_model=192, layers=4, type=gat, edge_features=[Δλ, mol_i, mol_j, seam]
fusion	type=concat_mlp, hidden=384, dropout=0.1
decoder_mu	scales=[64,32,283], aux_loss=true
head_sigma	type=flow, sigma_min=1e-4, quantiles=false


⸻

16) Tests
	•	Unit — symbolic loss components, encoders, decoders.
	•	Integration — selftest --deep, pipeline consistency checker.
	•	Smoke E2E — tiny planet set, predict + dashboard.

⸻

17) ADR & Change Control
	•	ADRs in docs/adr/ for contract changes.
	•	CI gate: all tests must pass before merge.
	•	Release tagging when producing submission candidates.

⸻

18) Operational Quick Start

# Verify environment
spectramind --version
spectramind selftest --deep

# Train curriculum
spectramind train --phase mae
spectramind train --phase supervised +loss.λ_sm=0.2 +model.fusion.type=cross-attend

# Predict + calibrate + bundle
spectramind predict --out-csv outputs/submission.csv
spectramind calibrate-temp
spectramind calibrate-corel
spectramind submit bundle --submission outputs/submission.csv --out-zip outputs/submission_bundle.zip

# Diagnostics
spectramind diagnose dashboard --html outputs/diagnostics/v1.html


⸻

19) Engineering Deep Dive

Contains:
	•	Full symbolic loss derivations.
	•	Event log JSONL schema.
	•	DVC DAG definition.
	•	CI job skeletons.