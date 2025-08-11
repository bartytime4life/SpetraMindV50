SpectraMind V50 — Authoritative Mission Architecture & Engineering Doctrine

Mission‑grade, neuro‑symbolic, physics‑informed AI system for the NeurIPS 2025 Ariel Data Challenge. This is the single source of truth for science, contracts, ops, and code.

⸻

0) Command Intent
	•	Win the leaderboard without gimmicks: physics‑faithful μ/σ, calibrated, robust OOD.
	•	Neuro‑symbolic by design: deep encoders + differentiable physics constraints + ex‑post logic audit.
	•	Reproducibility first: every artifact traceable to code SHA, config hash, env, seeds.
	•	Runtime disciplined: ≤9 h / ~1,100 planets (≈≤30 s/planet) under Kaggle GPU limits.
	•	Operator‑first explainability: SHAP × symbolic overlays, UMAP/t‑SNE, FFT/smoothness, coverage plots.

⸻

1) Scientific & Competitive Landscape

1.1 Observational Physics (scope of constraints)
	•	Transmission spectroscopy: wavelength‑dependent transit depth encodes atmospheric absorption.
	•	Scale height H=\frac{kT}{\mu g}: hotter/low‑g planets → deeper features.
	•	Stellar limb darkening: prefer 3D MHD‑derived laws; errors project into μ bias.
	•	Astrophysical noise: spots/faculae, oscillations; partially separable via FGS1 context.
	•	Instrument systematics: pointing jitter × intrapixel response, non‑linearity, ramps, seams.

1.2 Challenge Realities
	•	Primary metric: Gaussian Log‑Likelihood (GLL) over 283 bins (joint μ/σ).
	•	FGS1 dominance: ~58× weight vs a single AIRS bin → FGS1‑first modeling.
	•	OOD: atmospheres/conditions unseen; symbolic rules act as invariant priors.
	•	Budget: ≤9 h hard wall; enforce via telemetry + “fast kaggle” overrides.

⸻

2) End‑to‑End System (Level‑1 Dataflow)

flowchart LR
  subgraph RAW[Raw Inputs]
    A[FGS1 cubes]:::raw
    B[AIRS-CH0 cubes]:::raw
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
  classDef raw fill:#f6f6ff,stroke:#7a7ab8;


⸻

3) Data Contracts & Schemas

Invariant: never clip negatives in calibrated frames; always propagate variance; log correction magnitudes.

3.1 Directory Topology

/data
  raw/{fgs1,airs_ch0}/...
  calibrated/{fgs1,airs}/...
  features/{fgs1_white,airs_bins}/...
  splits/groupkfold.json

3.2 Calibrated Tensors
	•	fgs1_{planet}.npz
	•	keys: frames: float32[T,H,W], variance: float32[T,H,W], mask: bool[T,H,W]
	•	airs_{planet}.npz
	•	keys: frames, variance, mask, trace_meta: json (slit location, seam indices, dispersion poly)

3.3 Feature Tensors
	•	fgs1_white_{planet}.npz:
flux: float32[T], time: float64[T], centroid: float32[T,2], jitter: float32[T,2]
	•	airs_bins_{planet}.npz:
flux: float32[T,283], time: float64[T]

3.4 Submission Schema (schemas/submission.schema.json)
	•	Columns: mu_0..mu_282, sigma_0..sigma_282 (decimal strings OK; parse to float32).
	•	Validator: src/spectramind/utils/schema_check.py.

⸻

4) Calibration Kill Chain (Level‑2)

All steps DVC‑tracked. For each step k, log JSON: {"step": k, "t_ms": ..., "params": {...}, "delta_stats": {...}}.

	1.	ADC reversal / bias

	•	E = (R - o)\cdot g; \mathrm{Var}_E = g^2 \mathrm{Var}_R.
	•	Log: g, o, mean/std pre/post.

	2.	Bad pixel mask & interpolation

	•	Mask conditions: hot | cold | NaN | sat | persistence | cosmic.
	•	Interp: spatio‑temporal kernel with noise floor \epsilon^2:
E[\mathcal{M}] \leftarrow \text{interp}(E,\mathcal{M});\ \mathrm{Var}_E[\mathcal{M}] \mathrel{+}= \epsilon^2.
	•	Log: mask_frac, interp_kernel.

	3.	Non‑linearity correction

	•	Per‑pixel LUT/poly f: E_{\text{lin}} = f(E).
	•	Variance via Jacobian: \mathrm{Var}_{\text{lin}} = (f’(E))^2 \mathrm{Var}_E.
	•	Log: median correction (ppm), outlier clips.

	4.	Dark subtraction

	•	E_d = E_{\text{lin}} - D; \mathrm{Var}d = \mathrm{Var}{\text{lin}} + \mathrm{Var}_D.
	•	Log: dark level stats, temperature proxy if present.

	5.	Flat‑fielding

	•	E_f = E_d / F; \mathrm{Var}_f = \mathrm{Var}_d / F^2.
	•	Log: flat norm stats; cross‑hatch flag if known.

	6.	Trace extraction / photometry

	•	FGS1 white‑light: aperture sum + annulus background; var sums accordingly.
	•	AIRS spectral: optimal extraction along slit; seam handling with seam mask \mathcal{S}.
	•	Log: aperture radii, seam indices, throughput.

Outputs: calibrated frames → features (fgs1_white, airs_bins) + per‑stage logs.

⸻

5) Modeling System (Level‑2)

5.1 Encoders

FGS1 Mamba SSM
	•	Input x_{\text{fgs1}}\in\mathbb{R}^{B\times T\times C} (C: flux, jitter, centroid, etc.).
	•	Output h_{\text{fgs1}}\in\mathbb{R}^{B\times D} (+ optional per‑step latents for UMAP).
	•	Options: bidirectional, dropout, residual, AMP on.

AIRS GAT (physics‑informed graph)
	•	Nodes V=\{0..282\}; edges E=E_\lambda \cup E_{\text{mol}} \cup E_{\text{seam}}.
	•	Node features x_i\in \mathbb{R}^{C_n} (time‑stat features or encoder outputs).
	•	Edge features \(e_{ij}=[\Delta\lambda_{ij}, \text{mol\_tag\_i}, \text{mol\_tag\_j}, \text{seam\flag}{ij}]\).
	•	Output h_{\text{airs}}\in\mathbb{R}^{B\times D}.
	•	Export attention weights for diagnostics.

5.2 Fusion
	•	h = \text{Fuse}(h_{\text{fgs1}}, h_{\text{airs}}); types: concat+MLP | cross-attend | gate.
	•	Config: model.fusion.type, dim, dropout.

5.3 Decoders

Multi‑Scale μ Decoder
	•	Coarse‑to‑fine heads: predict \tilde{\mu}^{(k)} at scales k\in\{coarse, mid, fine\} with skip fusion.
	•	Final \mu = \tilde{\mu}^{(fine)}; optionally auxiliary losses at other scales.

σ Head (flow/quantile)
	•	Heteroscedastic \sigma = \text{Softplus}(g(h)) + \sigma_{\min}.
	•	Optional quantiles (q_{10}, q_{50}, q_{90}) with monotonic penalties.
	•	TorchScript‑safe; compatible with temp scaling + COREL.

⸻

6) Objective: Likelihood + Symbolic Physics

6.1 Gaussian Log‑Likelihood (primary)

For planet p, bins i\in[0,282], targets y_{p,i}, predictions \mu_{p,i}, \sigma_{p,i}:
\mathcal{L}{\text{gll}}(p)=\frac{1}{2}\sum{i}\Big[\log(2\pi\sigma_{p,i}^2)+\frac{(y_{p,i}-\mu_{p,i})^2}{\sigma_{p,i}^2}\Big]
Numerics: clamp \sigma\leftarrow\max(\sigma,\sigma_{\min}), stabilize with \log\sigma parameterization if needed.

6.2 Symbolic Pack (differentiable penalties)

Let \mu\in\mathbb{R}^{283}, wavelength‑ordered; \mathbb{1}[\cdot] mask; \odot elementwise.
	1.	Smoothness (2nd derivative L2)
\delta_i = \mu_{i+1}-2\mu_i+\mu_{i-1},\quad
\mathcal{L}{\text{sm}}=\sum{i=1}^{281} w^{\text{sm}}_i \,\delta_i^2

	•	Mask near edges; weight stronger in continuum windows or away from known sharp lines.

	2.	Non‑negativity
\mathcal{L}{\text{nn}}=\sum{i} w^{\text{nn}}_i\,\text{ReLU}(-\mu_i)

	•	Option: square ReLU for stronger push: \text{ReLU}(-\mu_i)^2.

	3.	Molecular coherence (soft implication)
Given molecule windows W_m (H₂O/CO₂/CH₄) and templates t^m\ge0:
s^m_i = \frac{\mu_i}{\max(\epsilon, \| \mu_{W_m}\|2)};\
\mathcal{L}{\text{coh}}^m=\sum_{i\in W_m} w^{m}_i\,\text{ReLU}(t^m_i - s^m_i)^2

	•	Intuition: if molecule m is present, normalized response should not fall below template envelope.

	4.	Seam continuity (across detector seam index s)
\mathcal{L}{\text{seam}} = w^{\text{seam}}\,(\mu{s^-}-\mu_{s^+})^2
	5.	Chemistry envelopes (CH₄:H₂O:CO₂ area ratios)
Let band integrals A_m=\sum_{i\in W_m}\mu_i\Delta\lambda_i. For bounds [r^{\min}{ab}, r^{\max}{ab}]:
\mathcal{L}{\text{ratio}}=\sum{(a,b)} \Big[\text{ReLU}\big(r^{\min}{ab}-\tfrac{A_a}{A_b+\epsilon}\big)+
\text{ReLU}\big(\tfrac{A_a}{A_b+\epsilon}-r^{\max}{ab}\big)\Big]
	6.	Quantile monotonicity (if predicting quantiles)
\mathcal{L}{\text{qm}}=\sum_i \Big[\text{ReLU}(q{10,i}-q_{50,i}) + \text{ReLU}(q_{50,i}-q_{90,i})\Big]

Total symbolic
\mathcal{L}{\text{sym}}=\lambda{\text{sm}}\mathcal{L}{\text{sm}}+\lambda{\text{nn}}\mathcal{L}{\text{nn}}+
\lambda{\text{coh}}\sum_m\mathcal{L}{\text{coh}}^m+\lambda{\text{seam}}\mathcal{L}{\text{seam}}+
\lambda{\text{ratio}}\mathcal{L}{\text{ratio}}+\lambda{\text{qm}}\mathcal{L}_{\text{qm}}

Total objective
\mathcal{L}=\mathcal{L}{\text{gll}} + \mathcal{L}{\text{sym}}

6.2.1 Vectorized PyTorch pseudocode (stable, mask‑aware)

# Inputs: mu: (B, 283), sigma: (B, 283), y: (B, 283), eps=1e-6
# Masks: valid: (283,), seam_idx: int, molecule windows dict m->LongTensor indices
# Templates: templates[m]: (|W_m|,) nonnegative; dl: (283,) bin widths

# GLL
sigma_clamped = sigma.clamp_min(1e-4)
res = (y - mu) / sigma_clamped
L_gll = 0.5 * ( (res**2) + (2.0 * sigma_clamped.log()) + math.log(2*math.pi) )
L_gll = (L_gll * valid).sum(dim=1).mean()

# Smoothness (central second difference)
d2 = mu[..., 2:] - 2*mu[..., 1:-1] + mu[..., :-2]
w_sm = smoothness_weights(valid[1:-1]).to(mu)  # e.g., stronger off-lines
L_sm = (w_sm * d2**2).sum(dim=1).mean()

# Non-negativity
L_nn = torch.relu(-mu).sum(dim=1).mean()

# Molecular coherence
L_coh = 0.0
for m, idx in W_m.items():
    mu_m = mu.index_select(dim=1, index=idx)
    t_m  = templates[m].to(mu)
    norm = mu_m.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    s_m  = mu_m / norm
    L_coh = L_coh + (torch.relu(t_m - s_m)**2).sum(dim=1).mean()

# Seam continuity
s = seam_idx
L_seam = ((mu[..., s-1] - mu[..., s])**2).mean()

# Area ratios
def band_area(idx):
    return (mu.index_select(1, idx) * dl.index_select(0, idx)).sum(dim=1)
areas = { m: band_area(W_m[m]) for m in W_m }
def ratio_penalty(a, b, rmin, rmax):
    r = areas[a] / (areas[b] + 1e-6)
    return (torch.relu(rmin - r) + torch.relu(r - rmax)).mean()
L_ratio = sum( ratio_penalty('ch4','h2o', rmin_ch4_h2o, rmax_ch4_h2o),
               ratio_penalty('co2','h2o', rmin_co2_h2o, rmax_co2_h2o) )

# Quantile monotonicity (if present)
if q10 is not None:
    L_qm = (torch.relu(q10 - q50) + torch.relu(q50 - q90)).sum(dim=1).mean()
else:
    L_qm = 0.0

L_sym = λ_sm*L_sm + λ_nn*L_nn + λ_coh*L_coh + λ_seam*L_seam + λ_ratio*L_ratio + λ_qm*L_qm
L_total = L_gll + L_sym


⸻

7) Uncertainty Calibration (post‑hoc)

7.1 Temperature Scaling
	•	Learn global \tau>0 minimizing val GLL: \sigma^\ast = \tau \sigma.
	•	1‑D line search or LBFGS; stable with \log \tau param.

7.2 COREL Spectral Conformalization
	•	Conformal score per bin i: \epsilon_i = \frac{|y_i-\mu_i|}{\sigma^\ast_i}.
	•	Pick quantile q_i to target coverage 1-\alpha on val; calibrated
\sigma’_i = q_i \sigma^\ast_i.
	•	Outputs: per‑bin q_i.csv, molecule‑region coverage plots, z‑score histograms.

⸻

8) Training Curriculum & Schedules
	1.	Phase MAE

	•	Masking policies: random | block | molecule‑band | SNR‑weighted.
	•	Decoder loss: MSE on masked bins; optional σ‑pretraining (predict log‑σ).

	2.	Phase Contrastive (optional)

	•	Positive: in‑transit segments; negative: out‑of‑transit; temperature jitter.
	•	InfoNCE on latents; UMAP separation expected.

	3.	Phase Supervised

	•	Objective \mathcal{L}=\text{GLL}+\lambda\cdot\text{Symbolic}.
	•	AMP, cosine LR, grad accumulation, checkpoint every N epochs.

	4.	Calibration

	•	Temp scaling → COREL; log coverage deltas, molecule summaries.

CV Protocol: GroupKFold on planet_id; no cross‑planet leakage.

⸻

9) CLI‑First MLOps & Config Discipline

9.1 Unified CLI (spectramind.py)

Subcommands (all write to logs + JSONL):
	•	train [--phase mae|contrastive|supervised] [--config ...]
	•	predict --out-csv outputs/submission.csv
	•	calibrate-temp / calibrate-corel
	•	diagnose [dashboard|gll|fft|umap|tsne]
	•	dashboard --html outputs/diagnostics/v{N}.html [--open]
	•	submit bundle --submission ... --out-zip ...
	•	selftest [--deep]
	•	ablate [--grid ... --top_n ... --md --open_html]

9.2 Hydra Config Tree (authoritative)

configs/
  config_v50.yaml            # root composition
  data/{local.yaml,kaggle.yaml}
  model/{fgs1_mamba.yaml,airs_gnn.yaml,decoder.yaml,sigma_head.yaml,fusion.yaml}
  train/{mae.yaml,contrastive.yaml,supervised.yaml}
  calib/{temp.yaml,corel.yaml}
  diag/{dashboard.yaml,umap.yaml,tsne.yaml,fft.yaml,shap.yaml}
  overrides/{kaggle_fast.yaml}
  experiment/{exp_baseline.yaml,exp_kaggle_fast.yaml}

9.3 Logging & Telemetry (canonical formats)

Human log v50_debug_log.md (append‑only, UTC ISO, zero usec)

### CLI: train
- ts: 2025-08-11T11:34:00Z
- sha: 1b2c3d4
- cfg_hash: abcd1234
- host: kaggle-01
- cuda: 12.1, driver 550.xx, torch 2.4
- seed: 1337
- phase: supervised
- notes: "λ_sm=0.2, fusion=cross-attend"

Machine log events.jsonl (one JSON per CLI call)

{"ts":"2025-08-11T11:34:00Z","cmd":"train","args":{"phase":"supervised"},"git_sha":"1b2c3d4","cfg_hash":"abcd1234","env":{"cuda":"12.1","driver":"550.xx","torch":"2.4"},"seed":1337,"run_id":"v50_20250811_113400_abcd"}


⸻

10) Diagnostics & Dashboard (operator tools)
	•	Back: FastAPI; Front: React + Tremor + D3.
	•	Artifacts embedded in HTML:
	•	UMAP/t‑SNE latents (interactive; planet hyperlinks; confidence shading).
	•	SHAP overlays × symbolic violation heatmaps.
	•	FFT power / smoothness spectra.
	•	COREL coverage plots; z‑score histograms; per‑bin qᵢ table.
	•	CLI log table; Config Hash × GLL leaderboard heatmap.
	•	Script: generate_html_report.py (versioned vN.html), tolerant of missing artifacts.

⸻

11) CI/CD Doctrine (merge gates)

11.1 Sequence (Mermaid)

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

11.2 Acceptance tests per phase

Phase	Gate	Artifact
0	--version, selftest	logs, events.jsonl
1	calibrate DVC	calibrated/ features/
2	encoders+symbolic unit tests	violation JSON
3	temp+COREL	coverage plots, qᵢ.csv
4	dashboard	diagnostics_vN.html
5	runtime budget	timing CSV, submission.zip


⸻

12) Reproducibility Stack
	•	Code/Config: Git SHA + Hydra config hash on every run.
	•	Env: Poetry lock; optional CUDA Docker for parity.
	•	Data/Models: DVC (or lakeFS) with immutable hashes; optional DVC Experiments.
	•	Provenance: run manifest run_hash_summary_v50.json (SHA, cfg, seeds, env, artifacts).

⸻

13) Failure Modes & Runbooks
	•	σ underestimates (overconfident) → run calibrate-temp; if molecule‑region failures persist, run calibrate-corel and inspect region coverage.
	•	Seam discontinuity spikes → increase λ_seam; verify seam indices; inspect AIRS slit trace; re‑flat if needed.
	•	FGS1 drift → expand Mamba width/depth; enable bidirectional; add limb‑darkening features to FGS1 channel.
	•	Runtime over budget → enable overrides/kaggle_fast.yaml (batch size, caching); reduce diagnostics in Kaggle mode.

⸻

14) Symbolic Logic Engine (ex‑post auditor)

14.1 Rule evaluation (per planet)
	•	Inputs: \mu, (q_{10},q_{50},q_{90}) optional, molecule masks W_m, seam index s.
	•	Outputs: constraint_violation_log.json with:

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


	•	Severity bands: ok, warn, fail thresholds configurable per rule.

14.2 Influence & overlays
	•	Compute \partial\mathcal{L}_{\text{sym}}/\partial \mu per bin for “symbolic influence maps”.
	•	Overlay on SHAP heatmaps in dashboard; highlight dominant rule per region.

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
	•	test_symbolic_smoothness_second_diff.py: synthetic stepped μ → high penalty; smooth μ → low.
	•	test_nonnegativity.py: negative μ bins penalized; zero gradient above 0.
	•	test_molecule_coherence.py: inject template inside W_m → near‑zero penalty.
	•	test_quantile_monotonicity.py: shuffled quantiles → positive penalty.
	•	Integration
	•	selftest.py --deep: checks CLI registration, file presence, dummy forward, TorchScript export.
	•	pipeline_consistency_checker.py: DVC DAG ↔ CLI map ↔ configs coherence.
	•	Smoke E2E: tiny planet set → predict + dashboard must run and emit artifacts.

⸻

17) ADR & Change Control
	•	ADRs in docs/adr/ for any module contract change (I/O shapes, rule definitions, masks).
	•	PR Gating: CI must be fully green; artifacts uploaded on each PR for review.
	•	Release Tagging: tag when a submission candidate (validated bundle + diagnostics) is produced.

⸻

18) Run Recipes (copy/paste)
	•	Quick verify

spectramind --version
spectramind selftest --deep


	•	Train (curriculum)

spectramind train --phase mae
spectramind train --phase supervised +loss.λ_sm=0.2 +model.fusion.type=cross-attend


	•	Predict + calibrate + bundle

spectramind predict --out-csv outputs/submission.csv
spectramind calibrate-temp
spectramind calibrate-corel
spectramind submit bundle --submission outputs/submission.csv --out-zip outputs/submission_bundle.zip


	•	Diagnostics (HTML)

spectramind diagnose dashboard --html outputs/diagnostics/v1.html



⸻

19) Appendix A — Symbolic Loss Pseudocode (module‑level)

symbolic_loss.forward(mu, extras)

Inputs:
	•	mu: (B, 283)
	•	extras: dict with valid_mask, seam_idx, W_m (molecule window indices), templates, dl, quantiles?

Outputs:
	•	loss_sym: Tensor
	•	components: Dict[str, float] (sm, nn, coh, seam, ratio, qm)
	•	grads_mu: Optional[(B, 283)] if requested for influence maps

Algorithm:
	1.	Compute masks, seam index; pre‑build weights (smoothness_weights) from valid_mask & line list.
	2.	L_sm: central second difference; mask edges; multiply by per‑bin weights; mean over batch.
	3.	L_nn: relu(-mu); mean over batch.
	4.	L_coh: for each molecule m:
	•	select bins; normalize by L2 norm; hinge on templates[m] - s_m.
	5.	L_seam: seam delta squared; mean.
	6.	L_ratio: band integrals via (mu * dl); ratio hinge per pair.
	7.	L_qm: if q10/q50/q90 present, apply monotonic hinge.
	8.	Return weighted sum + components dict (+ grads if create_graph).

Numerical Guards:
	•	Clamp norms with 1e-6; clamp σ_min; compute in FP32 even under AMP for stability.

⸻

20) Appendix B — Events JSONL Schema (for downstream tools)

{
  "ts": "ISO-8601 UTC",
  "run_id": "string",
  "cmd": "train|predict|diagnose|dashboard|calibrate-temp|calibrate-corel|submit|selftest|ablate",
  "args": { "k": "v", "...": "..." },
  "git_sha": "^[0-9a-f]{7,}$",
  "cfg_hash": "^[0-9a-f]{8}$",
  "env": { "cuda": "12.1", "driver": "550.xx", "torch": "2.4", "device": "GPU-0:XX GB" },
  "seed": 1337,
  "artifacts": ["path1", "path2"],
  "duration_ms": 12345,
  "status": "ok|warn|fail"
}


⸻

21) Appendix C — DVC DAG (indicative)

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


⸻

22) Appendix D — CI Hints (jobs skeleton)
	•	ci.yml jobs:
	•	setup: cache Poetry, set CUDA toolkit
	•	selftest: run spectramind selftest
	•	unit: pytest -q (symbolic + encoders + utils)
	•	smoke-e2e: tiny dataset → predict + dashboard
	•	consistency: pipeline_consistency_checker.py
	•	artifacts: upload CSV/HTML/logs
	•	gate: require all above green

⸻

23) Roadmap Milestones
	1.	Phase 0: repo hardening, CLI integrity, Hydra migration, CI skeleton.
	2.	Phase 1: calibration + features (DVC), timing/variance logs.
	3.	Phase 2: dual encoders, fusion, decoders, full symbolic pack, unit tests.
	4.	Phase 3: MAE → (contrastive) → supervised; temp+COREL calibration.
	5.	Phase 4: diagnostics HTML bundle + dashboard server.
	6.	Phase 5: retrieval models, HITRAN/ExoMol overlays, synthetic planets, advanced UQ.

⸻

Final note

This file is the contract. If a PR changes any of: data shapes, masks, rule math, seam index semantics, or CLI contracts, update this document, add an ADR, and make CI enforce the new invariants.