SpectraMind V50 — Mission Architecture & Engineering Doctrine

ESA Ariel × NeurIPS 2025 — Physics‑Informed, Neuro‑Symbolic AI for Exoplanet Spectroscopy

 

 

 


⸻

Executive Summary

SpectraMind V50 is a mission‑grade AI stack to win the NeurIPS 2025 Ariel Data Challenge with physics‑faithful μ/σ, robust OOD behavior, and operator‑first explainability. It combines:
	•	FGS1‑first temporal encoding (Mamba SSM) for high‑weight transit morphology.
	•	Physics‑informed spectral graph encoding (AIRS GAT) with λ‑adjacency, molecule bands, and seam edges.
	•	Symbolic physics constraints as differentiable losses and an ex‑post auditor.
	•	Post‑hoc uncertainty calibration (Temperature Scaling → COREL conformal).
	•	CLI‑first MLOps (Hydra configs, JSONL events, DVC/lakeFS, Poetry/Docker parity).

Targets: ≤ 9 h for ~1,100 planets on Kaggle; reproducible by SHA+config hash; rich diagnostics (UMAP/t‑SNE, SHAP×Symbolic, FFT, coverage).

⸻

Table of Contents
	1.	Scientific & Competitive Landscape
	2.	Level‑1 Dataflow
	3.	Data Contracts & Schemas
	4.	Calibration Kill Chain
	5.	Modeling System
	6.	Objective: Likelihood + Symbolic Physics
	7.	Uncertainty Calibration
	8.	Training Curriculum
	9.	MLOps & Reproducibility
	10.	Diagnostics & Dashboard
	11.	CI/CD Doctrine
	12.	Failure Modes & Runbooks
	13.	Symbolic Logic Engine
	14.	Exact Config Knobs
	15.	Tests
	16.	ADR & Change Control
	17.	Operational Quick Start
	18.	Appendices

⸻

Scientific & Competitive Landscape

Transmission Spectroscopy — transit depth vs wavelength encodes atmospheric absorption.
Scale Height

$$
H=\frac{kT}{\mu g}
$$

Hotter / lower‑g planets → deeper features.
FGS1 Dominance — broadband photometry anchors transits; ~58× weight vs any single AIRS bin.
AIRS‑CH0 — 283 spectral bins (1.95–3.90 µm) spanning H_2O/CH_4/CO_2 features.
Constraints — metric is Gaussian Log‑Likelihood (μ/σ); runtime ≤ 9 h; OOD atmospheres expected.

⸻

Level‑1 Dataflow

System overview (Mermaid):

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
    M2[AIRS GAT]
    MF[Fusion Layer]
    D1[μ Decoder (multi-scale)]
    D2[σ Head (flow/quantile)]
  end

  subgraph SYM[Symbolic Physics Engine]
    S1[Smoothness]
    S2[Non-negativity]
    S3[Molecular coherence]
    S4[Seam continuity]
    S5[Chemistry envelopes]
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

Data Contracts & Schemas

Invariant: never clip negatives in calibrated frames; always propagate variance; log correction magnitudes.

Directory topology

/data
  raw/
    fgs1/...
    airs_ch0/...
  calibrated/
    fgs1/...
    airs/...
  features/
    fgs1_white/...
    airs_bins/...
  splits/
    groupkfold.json

Calibrated tensors

File	Keys	Shape	Notes
fgs1_{planet}.npz	frames, variance, mask	(T,H,W)	float32/bool
airs_{planet}.npz	frames, variance, mask, trace_meta	(T,H,W)	slit position, seam indices, dispersion poly

Feature tensors
	•	fgs1_white_{planet}.npz → flux[T], time[T], centroid[T,2], jitter[T,2]
	•	airs_bins_{planet}.npz → flux[T,283], time[T]

Submission schema
	•	Columns: mu_0..mu_282, sigma_0..sigma_282 (decimal strings OK; parse to float32).
	•	Validator: src/spectramind/utils/schema_check.py.

⸻

Calibration Kill Chain

Schematic

graph TD
  A[ADC Reversal] --> B[Bad Pixel Mask/Interp]
  B --> C[Non-Linearity Correction]
  C --> D[Dark Subtraction]
  D --> E[Flat-Fielding]
  E --> F[Trace Extraction / Photometry]

Per‑step math & logs

1) ADC reversal / bias
E=(R-o)\cdot g,\ \ \mathrm{Var}_E=g^2\mathrm{Var}_R. Log: g, o, pre/post stats.

2) Bad pixel mask & interpolation
Mask: hot|cold|NaN|sat|persistence|cosmic. Interp with noise floor \epsilon^2. Log: mask_frac, interp_kernel.

3) Non‑linearity correction
Per‑pixel LUT/poly E_{lin}=f(E). Propagate \mathrm{Var}_{lin}=(f’(E))^2\mathrm{Var}_E. Log: median correction (ppm), clips.

4) Dark subtraction
E_d=E_{lin}-D,\ \ \mathrm{Var}d=\mathrm{Var}{lin}+\mathrm{Var}_D. Log: dark stats, temperature proxy.

5) Flat‑fielding
E_f=E_d/F,\ \ \mathrm{Var}_f=\mathrm{Var}_d/F^2. Log: flat normalization; cross‑hatch flag.

6) Trace extraction / photometry
FGS1: aperture sum + annulus background. AIRS: optimal extraction, seam‑aware. Log: aperture radii, seam indices, throughput.

⸻

Modeling System

Encoders
	•	FGS1 Mamba SSM — long‑sequence temporal encoder (bidir option, AMP).
Input x_{fgs1}\in\mathbb{R}^{B\times T\times C} → latent h_{fgs1}\in\mathbb{R}^{B\times D} (+ step latents for UMAP).
	•	AIRS GAT (physics‑informed) — nodes V=\{0..282\}; edges E_\lambda\cup E_{mol}\cup E_{seam}.
Node feat x_i\in\mathbb{R}^{C_n}; edge feat [\Delta\lambda_{ij},mol_i,mol_j,seam_{ij}].
Output h_{airs}\in\mathbb{R}^{B\times D}; attention exported.

Fusion — concat+MLP | cross‑attend | gated (configurable): h = \mathrm{Fuse}(h_{fgs1}, h_{airs}).

Decoders
	•	Multi‑scale μ — coarse→mid→fine with skip fusion; auxiliary losses optional.
	•	σ head (flow/quantile) — heteroscedastic \sigma=\mathrm{Softplus}(g(h))+\sigma_{min}; TorchScript‑safe; works with temp scaling & COREL.

⸻

Objective: Likelihood + Symbolic Physics

Primary (GLL)
$$
\mathcal{L}{\mathrm{gll}}=\frac{1}{2}\sum{i}\Big[\log(2\pi\sigma_i^2)+\frac{(y_i-\mu_i)^2}{\sigma_i^2}\Big]
$$
Numerics: clamp \sigma\ge\sigma_{min}; optional \log\sigma parameterization.

Symbolic pack (mask‑aware over 283 wavelength‑ordered bins)
	•	Smoothness (2nd diff L2) — \delta_i=\mu_{i+1}-2\mu_i+\mu_{i-1}, \mathcal{L}_{sm}=\sum w^{sm}_i\delta_i^2
	•	Non‑negativity — \mathcal{L}_{nn}=\sum w^{nn}_i\operatorname{ReLU}(-\mu_i)
	•	Molecular coherence — normalized response ≥ molecule templates t^m within windows W_m
	•	Seam continuity — \mathcal{L}{seam}=w^{seam}(\mu{s^-}-\mu_{s^+})^2
	•	Chemistry envelopes (area ratios) — A_m=\sum_{i\in W_m}\mu_i\Delta\lambda_i, hinge on A_a/(A_b+\epsilon) bounds
	•	Quantile monotonicity — if q_{10},q_{50},q_{90} present: hinge to keep q_{10}\le q_{50}\le q_{90}

Total
$$
\mathcal{L}=\mathcal{L}{\mathrm{gll}}
+\lambda{sm}\mathcal{L}{sm}
+\lambda{nn}\mathcal{L}{nn}
+\lambda{coh}!\sum_m!\mathcal{L}^m_{coh}
+\lambda_{seam}\mathcal{L}{seam}
+\lambda{ratio}\mathcal{L}{ratio}
+\lambda{qm}\mathcal{L}_{qm}
$$

⸻

Uncertainty Calibration

1) Temperature Scaling — learn global \tau>0 minimizing val GLL; \(\sigma^\=\tau\sigma\).
2) COREL Spectral Conformal — scores \epsilon_i=\frac{|y_i-\mu_i|}{\sigma^\_i}; choose quantile q_i for target coverage 1-\alpha; \(\sigma’_i=q_i\sigma^\*_i\).

Artifacts: per‑bin q_i.csv, coverage plots, z‑score histograms.

⸻

Training Curriculum

Phase 1 — MAE: physics‑aware masking; optional \log\sigma pretrain.
Phase 2 — Contrastive (opt.): InfoNCE on in‑transit vs out‑of‑transit latents.
Phase 3 — Supervised: \mathcal{L}=\text{GLL}+\lambda\cdot\text{Symbolic}; AMP, cosine LR, grad accumulation.
Phase 4 — Calibration: Temp scaling → COREL; log coverage deltas & molecule summaries.

CV protocol: GroupKFold on planet_id; no cross‑planet leakage.

⸻

MLOps & Reproducibility

Unified CLI (Typer)

spectramind --version
spectramind selftest --deep
spectramind train --phase mae|contrastive|supervised [overrides...]
spectramind predict --out-csv outputs/submission.csv
spectramind calibrate-temp
spectramind calibrate-corel
spectramind diagnose dashboard --html outputs/diagnostics/v1.html [--open]
spectramind submit bundle --submission outputs/submission.csv --out-zip outputs/submission_bundle.zip

Hydra config tree

configs/
  config_v50.yaml
  data/{local.yaml,kaggle.yaml}
  model/{fgs1_mamba.yaml,airs_gnn.yaml,decoder.yaml,sigma_head.yaml,fusion.yaml}
  train/{mae.yaml,contrastive.yaml,supervised.yaml}
  calib/{temp.yaml,corel.yaml}
  diag/{dashboard.yaml,umap.yaml,tsne.yaml,fft.yaml,shap.yaml}
  overrides/{kaggle_fast.yaml}
  experiment/{exp_baseline.yaml,exp_kaggle_fast.yaml}

Telemetry & provenance
	•	Human log v50_debug_log.md (UTC ISO, append‑only).
	•	Machine log events.jsonl (one JSON per CLI call).
	•	Run manifest run_hash_summary_v50.json (git SHA, cfg hash, seeds, env, artifacts).

⸻

Diagnostics & Dashboard
	•	Backend: FastAPI • Frontend: React + Tremor + D3
	•	Views: UMAP/t‑SNE, SHAP×Symbolic overlays, FFT spectra, COREL coverage & q_i table, Config‑Hash × GLL heatmap, CLI log table.
	•	Generator: generate_html_report.py → outputs/diagnostics/vN.html (graceful when some artifacts are missing).

⸻

CI/CD Doctrine

PR gate sequence (Mermaid)

sequenceDiagram
  participant Dev
  participant CI as GitHub Actions
  Dev->>CI: Push PR
  CI->>CI: Poetry/Docker setup
  CI->>CI: selftest.py (fast)
  CI->>CI: unit tests (symbolic, encoders, decoders)
  CI->>CI: smoke E2E (predict + dashboard)
  CI->>CI: pipeline_consistency_checker.py
  CI->>Dev: Upload artifacts (CSV/HTML/logs)
  alt all green
    CI->>Dev: Merge allowed
  else any red
    CI->>Dev: PR blocked
  end

Acceptance artifacts

Phase	Gate	Artifact
0	--version, selftest	logs, events.jsonl
1	DVC calibrate	calibrated/, features/
2	unit tests	violation JSON
3	temp + COREL	coverage plots, q_i.csv
4	dashboard	diagnostics_vN.html
5	runtime budget	timing CSV, submission.zip


⸻

Failure Modes & Runbooks
	•	Overconfident σ → calibrate-temp; if molecule bands under‑covered, run calibrate-corel.
	•	Seam discontinuity → ↑ λ_seam; verify seam index; re‑flat or re‑extract if needed.
	•	FGS1 drift → increase Mamba width/depth; enable bidir; add limb‑darkening features.
	•	Runtime over budget → -c overrides/kaggle_fast.yaml, ↑ batches, skip heavy diagnostics during train/predict.

⸻

Symbolic Logic Engine

Per‑planet auditor comparing \mu (and optional quantiles) to constraints; writes constraint_violation_log.json and influence maps \partial\mathcal{L}_{sym}/\partial\mu overlayed on SHAP.

{
  "planet_id": "P123",
  "violations": {
    "smoothness": {"score": 0.012, "top_bins": [101,102]},
    "nonneg": {"count": 0, "min": 0.0},
    "coherence:H2O": {"score": 0.004, "bins":[12,13,14]},
    "seam": {"delta": 0.00032},
    "ratios": {"ch4_h2o": {"r": 0.18, "status": "ok"}},
    "quantile_monotonicity": {"count": 0}
  }
}


⸻

Exact Config Knobs

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
  temp:  { enabled: true, tau_init: 1.0 }
  corel: { enabled: true, alpha: 0.1 }
diag:
  html: { open: false, out: "outputs/diagnostics/v1.html" }
overrides:
  kaggle_fast: { batch_predict: 2048, skip_heavy_diag: true }


⸻

Tests

Unit — smoothness 2nd‑diff, non‑negativity, molecular coherence, quantile monotonicity, encoder/decoder IO.
Integration — selftest --deep, TorchScript export, pipeline consistency checker.
Smoke E2E — tiny planet subset: predict → dashboard (artifacts emitted).

⸻

ADR & Change Control
	•	ADRs under docs/adr/ for I/O shapes, rule math, masks, seam semantics, CLI contracts.
	•	Merge gates: CI must be green; artifacts uploaded for review.
	•	Release tags: when producing a submission candidate (validated bundle + diagnostics).

⸻

Operational Quick Start

Verify

spectramind --version
spectramind selftest --deep

Train (curriculum)

spectramind train --phase mae
spectramind train --phase supervised +loss.λ_sm=0.2 +model.fusion.type=cross-attend

Predict + calibrate + bundle

spectramind predict --out-csv outputs/submission.csv
spectramind calibrate-temp
spectramind calibrate-corel
spectramind submit bundle --submission outputs/submission.csv --out-zip outputs/submission_bundle.zip

Diagnostics

spectramind diagnose dashboard --html outputs/diagnostics/v1.html


⸻

Appendices

<details>
<summary><b>Appendix A — Symbolic Loss (Vectorized, mask‑aware PyTorch)</b></summary>


# mu, sigma, y: (B, 283) ; valid: (283,)
# W_m: dict[str, LongTensor]; templates: dict[str, Tensor]; dl: (283,)
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

</details>


<details>
<summary><b>Appendix B — Events JSONL Schema</b></summary>


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

</details>


<details>
<summary><b>Appendix C — DVC DAG (indicative)</b></summary>


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

</details>
