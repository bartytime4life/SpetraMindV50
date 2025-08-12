⸻

SpectraMind V50 — Mission Architecture & Engineering Doctrine

ESA Ariel × NeurIPS 2025 Challenge – Physics-Informed, Neuro-Symbolic AI for Exoplanet Spectroscopy

⸻

Executive Summary

SpectraMind V50 is a mission-grade, neuro‑symbolic, physics‑informed AI system engineered to excel in the NeurIPS 2025 Ariel Data Challenge while setting a standard for scientific fidelity, reproducibility, and operator‑first explainability.

It fuses:
	•	FGS1‑first temporal encoding (Mamba SSM) for high‑weight transit morphology.
	•	Physics‑informed spectral graph encoding (AIRS GAT) with λ‑adjacency, molecular windows, and seam edges.
	•	Symbolic physics constraints as differentiable penalties and an ex‑post auditing engine.
	•	Uncertainty calibration (Temperature Scaling + COREL spectral conformal) aligned to the GLL metric.
	•	CLI‑first MLOps with Hydra configs, Poetry/Docker parity, DVC/lakeFS data control, and JSONL telemetry.
	•	Local LLM hooks for developer/operator assistance and explainability workflows.

Designed outcomes:
	•	≤ 9 h for ~1,100 planets on Kaggle GPUs.
	•	Physically plausible μ/σ spectra robust to OOD atmospheres.
	•	Interactive diagnostics: SHAP × symbolic overlays, UMAP/t‑SNE, FFT smoothness, coverage plots.

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
	9.	CLI-First MLOps & Reproducibility Stack
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

Observational Physics
	•	Transmission spectroscopy—wavelength‑dependent transit depth encodes atmospheric absorption.
	•	Scale height:
H = \frac{kT}{\mu g}
Hotter or low‑g planets → deeper features.
	•	Stellar limb darkening—prefer 3D MHD‑derived laws; errors project into μ bias.
	•	Astrophysical noise—spots/faculae, oscillations; separable via FGS1 context.
	•	Instrument systematics—pointing jitter, intrapixel response, non‑linearity, ramps, seams.

Challenge Realities
	•	Metric: Gaussian Log‑Likelihood (GLL) over 283 bins (joint μ/σ).
	•	FGS1 dominance: ~58× weight vs a single AIRS bin → FGS1‑first modeling.
	•	OOD: atmospheres/conditions unseen; symbolic rules act as invariant priors.
	•	Budget: ≤ 9 h hard wall; enforced by telemetry + Kaggle fast overrides.

⸻

2) End-to-End System (Level‑1 Dataflow)

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
    MF[Fusion]
    D1[Multi-Scale μ Decoder]
    D2[Flow/Quantile σ Head]
  end

  subgraph SYM[Symbolic Physics Engine]
    S1[Smoothness ∂²μ]
    S2[Non-negativity μ≥0]
    S3[Molecule coherence]
    S4[Seam continuity]
    S5[CH4:H2O:CO2 envelopes]
    S6[Quantile monotonicity]
  end

  subgraph CAL2[Uncertainty Calibration]
    T1[Temperature scaling]
    T2[COREL spectral conformal]
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

Calibrated Tensors

File	Keys	Shape	Notes
fgs1_{planet}.npz	frames, variance, mask	(T,H,W)	float32/bool
airs_{planet}.npz	frames, variance, mask, trace_meta	(T,H,W)	slit, seam, dispersion poly

Feature Tensors
	•	fgs1_white_{planet}.npz: flux[T], time[T], centroid[T,2], jitter[T,2]
	•	airs_bins_{planet}.npz: flux[T,283], time[T]

Submission Schema
	•	Columns: mu_0..mu_282, sigma_0..sigma_282 (decimal strings OK; parse to float32).
	•	Validator: src/spectramind/utils/schema_check.py.

⸻

4) Calibration Kill Chain (Level‑2)

All steps DVC‑tracked. For each step k, log JSON: {step, t_ms, params, delta_stats}.
	1.	ADC reversal / bias
E = (R - o)\cdot g;\quad \mathrm{Var}_E = g^2 \mathrm{Var}_R
	2.	Bad pixel mask & interpolation
Mask = hot|cold|NaN|sat|persistence|cosmic; add noise floor ε² for interpolants.
	3.	Non‑linearity correction
Per‑pixel LUT/poly E_{lin} = f(E), propagate \mathrm{Var}_{lin} = (f’(E))^2 \mathrm{Var}_E.
	4.	Dark subtraction
E_d = E_{lin} - D;\quad \mathrm{Var}d = \mathrm{Var}{lin} + \mathrm{Var}_D
	5.	Flat‑fielding
E_f = E_d / F;\quad \mathrm{Var}_f = \mathrm{Var}_d / F^2
	6.	Trace extraction / photometry
FGS1 white‑light aperture sum; AIRS optimal extraction along slit; seam mask.

Outputs: calibrated frames → features (fgs1_white, airs_bins) + per‑stage logs.

⸻

5) Modeling System (Level‑2)

FGS1 Mamba SSM
	•	Input x_{fgs1}\in\mathbb{R}^{B\times T\times C} (flux, jitter, centroid, …)
	•	Output h_{fgs1}\in\mathbb{R}^{B\times D} (+ optional per‑step latents for UMAP)
	•	Options: bidirectional, dropout, residual, AMP.

AIRS GAT (physics‑informed graph)
	•	Nodes V=\{0..282\}; edges E = E_\lambda \cup E_{mol} \cup E_{seam}
	•	Node features x_i\in\mathbb{R}^{C_n}; edge features [\Delta\lambda_{ij},\ \text{mol}_i,\ \text{mol}j,\ \text{seam}{ij}]
	•	Output h_{airs}\in\mathbb{R}^{B\times D}; attention weights exported for diagnostics.

Fusion

h = Fuse(h_fgs1, h_airs) — concat+MLP | cross‑attend | gated.

Decoders
	•	Multi‑Scale μ — coarse→mid→fine with skip fusion; auxiliary losses optional.
	•	σ Head — heteroscedastic \sigma=\mathrm{Softplus}(g(h))+\sigma_{min}; Flow or quantile variant.

⸻

6) Objective: Likelihood + Symbolic Physics

6.1 Gaussian Log‑Likelihood (primary)

\mathcal{L}{gll}(p)=\frac{1}{2}\sum{i}\left[\log(2\pi\sigma_{p,i}^2)+\frac{(y_{p,i}-\mu_{p,i})^2}{\sigma_{p,i}^2}\right]
Numerics: clamp \sigma\ge\sigma_{min}; \log \sigma parameterization as needed.

6.2 Symbolic Pack (differentiable penalties)

Let \mu\in\mathbb{R}^{283}, wavelength‑ordered.
	•	Smoothness (2nd derivative L2)
\delta_i=\mu_{i+1}-2\mu_i+\mu_{i-1}, \ \mathcal{L}_{sm}=\sum w^{sm}_i\delta_i^2
	•	Non‑negativity
\mathcal{L}_{nn}=\sum w^{nn}_i\,\mathrm{ReLU}(-\mu_i)
	•	Molecular coherence (soft implication)
Normalized response should not fall below molecule templates t^m\ge0 over windows W_m
	•	Seam continuity
\mathcal{L}{seam}=w^{seam}\left(\mu{s^-}-\mu_{s^+}\right)^2
	•	Chemistry envelopes (area ratios)
Band integrals A_m=\sum_{i\in W_m}\mu_i\Delta\lambda_i; hinge on ratio bounds
	•	Quantile monotonicity (if predicting quantiles)
\mathcal{L}{qm}=\sum_i[\mathrm{ReLU}(q{10,i}-q_{50,i})+\mathrm{ReLU}(q_{50,i}-q_{90,i})]

Total:
\mathcal{L}=\mathcal{L}{gll}+\lambda{sm}\mathcal{L}{sm}+\lambda{nn}\mathcal{L}{nn}
+\lambda{coh}\sum_m\mathcal{L}^m_{coh}+\lambda_{seam}\mathcal{L}{seam}
+\lambda{ratio}\mathcal{L}{ratio}+\lambda{qm}\mathcal{L}_{qm}

Vectorized PyTorch pseudocode (mask‑aware) – omitted here for brevity in the doc body; keep in Appendix A.

⸻

7) Uncertainty Calibration (post‑hoc)
	1.	Temperature Scaling — learn global \tau>0 minimizing val GLL: \(\sigma^\*\!=\tau\sigma\).
	2.	COREL Spectral Conformalization — per‑bin scores \epsilon_i=\frac{|y_i-\mu_i|}{\sigma^\_i}; pick quantile q_i for target coverage 1-\alpha; calibrated \sigma’_i=q_i\sigma^\_i.

Artifacts: per‑bin q_i.csv, coverage plots, z‑score histograms.

⸻

8) Training Curriculum & Schedules
	1.	MAE — physics‑aware masking; optional σ‑pretrain (predict log‑σ).
	2.	Contrastive (optional) — InfoNCE on in‑transit vs out‑of‑transit latents.
	3.	Supervised — \mathcal{L}=\text{GLL}+\lambda\cdot\text{Symbolic}; AMP, cosine LR, grad accumulation.
	4.	Calibration — Temp scaling → COREL; log coverage deltas.

CV protocol: GroupKFold on planet_id; no cross‑planet leakage.

⸻

9) CLI‑First MLOps & Reproducibility Stack

Unified CLI (spectramind.py)
	•	train, predict, calibrate-temp, calibrate-corel, diagnose, dashboard, submit, selftest, ablate.

Hydra Configs

configs/
  config_v50.yaml
  data/{local.yaml,kaggle.yaml}
  model/{fgs1_mamba.yaml,airs_gnn.yaml,decoder.yaml,sigma_head.yaml,fusion.yaml}
  train/{mae.yaml,contrastive.yaml,supervised.yaml}
  calib/{temp.yaml,corel.yaml}
  diag/{dashboard.yaml,umap.yaml,tsne.yaml,fft.yaml,shap.yaml}
  overrides/{kaggle_fast.yaml}
  experiment/{exp_baseline.yaml,exp_kaggle_fast.yaml}

Logging & Telemetry
	•	Human log v50_debug_log.md (UTC ISO, append‑only).
	•	Machine log events.jsonl (JSONL event stream).
	•	Run manifest run_hash_summary_v50.json (SHA, cfg, seeds, env, artifacts).

⸻

10) Diagnostics & Dashboard
	•	Backend: FastAPI; Front: React + Tremor + D3.
	•	Embedded artifacts: UMAP/t‑SNE, SHAP × symbolic overlays, FFT spectra, COREL coverage, z‑score histograms, CLI log table, Config‑Hash × GLL heatmap.
	•	Script: generate_html_report.py → outputs/diagnostics/vN.html.

⸻

11) CI/CD Doctrine (merge gates)

sequenceDiagram
  participant Dev
  participant CI as GitHub Actions
  Dev->>CI: Push PR
  CI->>CI: Poetry/Docker setup
  CI->>CI: selftest.py (fast)
  CI->>CI: unit tests (symbolic_loss, encoders, decoders)
  CI->>CI: smoke E2E (predict+dashboard)
  CI->>CI: pipeline_consistency_checker.py
  CI->>Dev: Upload artifacts (CSV, HTML, logs)
  alt all green
    CI->>Dev: Merge allowed
  else any red
    CI->>Dev: PR blocked
  end

Acceptance tests per phase

Phase	Gate	Artifact
0	--version, selftest	logs, events.jsonl
1	DVC calibrate	calibrated/, features/
2	unit tests	violation JSON
3	temp+COREL	coverage plots, q_i.csv
4	dashboard	diagnostics_vN.html
5	runtime budget	timing CSV, submission.zip


⸻

12) Reproducibility Stack
	•	Code/Config: Git SHA + Hydra config hash on every run.
	•	Env: Poetry lock; optional CUDA Docker for parity.
	•	Data/Models: DVC/lakeFS with immutable hashes; optional DVC Experiments.
	•	Provenance: run_hash_summary_v50.json.

⸻

13) Failure Modes & Runbooks
	•	σ underestimates (overconfident) → calibrate-temp; then calibrate-corel.
	•	Seam discontinuity → ↑\lambda_{seam}; verify seam indices; inspect AIRS slit trace; re‑flat.
	•	FGS1 drift → ↑ Mamba width/depth; enable bidirectional; add limb‑darkening features.
	•	Runtime over budget → overrides/kaggle_fast.yaml; reduce diagnostics in Kaggle mode.

⸻

14) Symbolic Logic Engine (ex‑post auditor)

Per planet output constraint_violation_log.json:

{
  "planet_id": "P123",
  "violations": {
    "smoothness": {"score": 0.012, "top_bins": [101,102]},
    "nonneg": {"count": 0, "min": 0.0},
    "coherence:H2O": {"score": 0.004, "bins":[...]},
    "seam": {"delta": 3.2e-4},
    "ratios": {"ch4_h2o": {"r": 0.18, "status": "ok"}},
    "quantile_monotonicity": {"count": 0}
  }
}

Severity: ok|warn|fail. Influence maps: \partial\mathcal{L}_{sym}/\partial \mu overlaid on SHAP.

⸻

15) Exact Config Knobs (minimal, but enough)

model:
  fgs1_mamba: { d_model: 256, n_layers: 8, bidir: true, dropout: 0.1 }
  airs_gnn:
    d_model: 192
    layers: 4
    type: gat
    edge_features: [delta_lambda, mol_i, mol_j, seam]
  fusion: { type: "concat_mlp", hidden: 384, dropout: 0.1 }
  decoder_mu: { scales: [64, 32, 283], aux_loss: true }
  head_sigma: { type: "flow", sigma_min: 1e-4, quantiles: false }
train:
  optim: { name: adamw, lr: 2.5e-4, wd: 0.01 }
  sched: { name: cosine, warmup_steps: 2000 }
  amp: true
  grad_accum: 2
  ckpt_every_epochs: 1
loss:
  λ_sm: 0.2
  λ_nn: 0.05
  λ_coh: 0.1
  λ_seam: 0.05
  λ_ratio: 0.05
  λ_qm: 0.02
calib:
  temp: { enabled: true, tau_init: 1.0 }
  corel: { enabled: true, alpha: 0.1 }
diag:
  html: { open: false, out: "outputs/diagnostics/v1.html" }
overrides:
  kaggle_fast: { batch_predict: 2048, skip_heavy_diag: true }


⸻

16) Tests (unit/integration)
	•	Unit
	•	test_symbolic_smoothness_second_diff.py: stepped μ → high penalty; smooth μ → low.
	•	test_nonnegativity.py: negative μ penalized; zero grad above 0.
	•	test_molecule_coherence.py: template inside W_m → near‑zero penalty.
	•	test_quantile_monotonicity.py: shuffled quantiles → positive penalty.
	•	Integration
	•	selftest.py --deep: CLI registration, file presence, dummy forward, TorchScript export.
	•	pipeline_consistency_checker.py: DVC DAG ↔ CLI map ↔ configs coherence.
	•	Smoke E2E
	•	tiny planet set → predict + dashboard must run and emit artifacts.

⸻

17) ADR & Change Control
	•	ADRs in docs/adr/ for any module contract change (I/O shapes, rule definitions, masks).
	•	PR gating: CI must be green; artifacts uploaded per PR.
	•	Tag release when a submission candidate (validated bundle + diagnostics) is produced.

⸻

18) Operational Quick Start

# Verify
spectramind --version
spectramind selftest --deep

# Train
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

19) Engineering Deep Dive (Appendices)

Appendix A — Symbolic Loss Pseudocode (Vectorized)

# Inputs: mu: (B, 283), sigma: (B, 283), y: (B, 283)
# Masks: valid: (283,), seam_idx: int, W_m: dict[str, LongTensor]
# Templates: templates[m]: (|W_m|,), dl: (283,)
sigma_clamped = sigma.clamp_min(1e-4)
res = (y - mu) / sigma_clamped
L_gll = 0.5 * ((res**2) + (2.0 * sigma_clamped.log()) + math.log(2*math.pi))
L_gll = (L_gll * valid).sum(dim=1).mean()

d2 = mu[..., 2:] - 2*mu[..., 1:-1] + mu[..., :-2]
w_sm = smoothness_weights(valid[1:-1]).to(mu)
L_sm = (w_sm * d2**2).sum(dim=1).mean()

L_nn = torch.relu(-mu).sum(dim=1).mean()

L_coh = 0.0
for m, idx in W_m.items():
    mu_m = mu.index_select(1, idx)
    t_m  = templates[m].to(mu)
    norm = mu_m.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    s_m  = mu_m / norm
    L_coh = L_coh + (torch.relu(t_m - s_m)**2).sum(dim=1).mean()

s = seam_idx
L_seam = ((mu[..., s-1] - mu[..., s])**2).mean()

def band_area(idx):
    return (mu.index_select(1, idx) * dl.index_select(0, idx)).sum(dim=1)
areas = {m: band_area(W_m[m]) for m in W_m}

def ratio_penalty(a, b, rmin, rmax):
    r = areas[a] / (areas[b] + 1e-6)
    return (torch.relu(rmin - r) + torch.relu(r - rmax)).mean()

L_ratio = ratio_penalty('ch4','h2o', rmin_ch4_h2o, rmax_ch4_h2o) \
        + ratio_penalty('co2','h2o', rmin_co2_h2o, rmax_co2_h2o)

L_qm = 0.0 if q10 is None else (torch.relu(q10 - q50) + torch.relu(q50 - q90)).sum(dim=1).mean()

L_sym = λ_sm*L_sm + λ_nn*L_nn + λ_coh*L_coh + λ_seam*L_seam + λ_ratio*L_ratio + λ_qm*L_qm
L_total = L_gll + L_sym

Appendix B — Events JSONL Schema

{
  "ts": "ISO-8601 UTC",
  "run_id": "string",
  "cmd": "train|predict|diagnose|dashboard|calibrate-temp|calibrate-corel|submit|selftest|ablate",
  "args": { "k": "v" },
  "git_sha": "^[0-9a-f]{7,}$",
  "cfg_hash": "^[0-9a-f]{8}$",
  "env": { "cuda": "12.1", "driver": "550.xx", "torch": "2.4", "device": "GPU-0:XX GB" },
  "seed": 1337,
  "artifacts": ["path1", "path2"],
  "duration_ms": 12345,
  "status": "ok|warn|fail"
}

Appendix C — DVC DAG (indicative)

stages:
  calibrate:
    cmd: python -m spectramind.calibration.run --in data/raw --out data/calibrated
    deps: [src/spectramind/calibration/*, data/raw]
    outs: [data/calibrated]
  features:
    cmd: python -m spectramind.features.run --in data/calibrated --out data/features
    deps: [src/spectramind/features/*, data/calibrated]
    outs: [data/features]
  train:
    cmd: spectramind train --phase supervised
    deps: [src/spectramind/models/*, src/spectramind/training/*, data/features]
    outs: [outputs/checkpoints]
  predict:
    cmd: spectramind predict --out-csv outputs/submission.csv
    deps: [outputs/checkpoints, data/features]
    outs: [outputs/submission.csv]
  diagnose:
    cmd: spectramind diagnose dashboard --html outputs/diagnostics/v1.html
    deps: [outputs/submission.csv]
    outs: [outputs/diagnostics/v1.html]

Appendix D — CI Hints (jobs skeleton)
	•	setup: cache Poetry, set CUDA toolkit
	•	selftest: spectramind selftest
	•	unit: pytest -q (symbolic + encoders + utils)
	•	smoke-e2e: tiny dataset → predict + dashboard
	•	consistency: pipeline_consistency_checker.py
	•	artifacts: upload CSV/HTML/logs
	•	gate: require all green

⸻