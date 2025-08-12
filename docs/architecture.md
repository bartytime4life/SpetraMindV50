**SpectraMind V50 — Authoritative Mission Architecture & Engineering Doctrine**

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
E = (R - o)\cdot g; \mathrm{Var}_E = g^2 \mathrm{Var}_R.
Log: g, o, mean/std pre/post.
	2.	Bad pixel mask & interpolation
Mask conditions: hot | cold | NaN | sat | persistence | cosmic.
Interp: spatio‑temporal kernel with noise floor \epsilon^2:
E[\mathcal{M}] \leftarrow \text{interp}(E,\mathcal{M});\ \mathrm{Var}_E[\mathcal{M}] \mathrel{+}= \epsilon^2.
Log: mask_frac, interp_kernel.
	3.	Non‑linearity correction
Per‑pixel LUT/poly f: E_{\text{lin}} = f(E).
Variance via Jacobian: \mathrm{Var}_{\text{lin}} = (f’(E))^2 \mathrm{Var}_E.
Log: median correction (ppm), outlier clips.
	4.	Dark subtraction
E_d = E_{\text{lin}} - D; \mathrm{Var}d = \mathrm{Var}{\text{lin}} + \mathrm{Var}_D.
Log: dark level stats, temperature proxy if present.
	5.	Flat‑fielding
E_f = E_d / F; \mathrm{Var}_f = \mathrm{Var}_d / F^2.
Log: flat norm stats; cross‑hatch flag if known.
	6.	Trace extraction / photometry
	•	FGS1 white‑light: aperture sum + annulus background; var sums accordingly.
	•	AIRS spectral: optimal extraction along slit; seam handling with seam mask \mathcal{S}.
Log: aperture radii, seam indices, throughput.

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
L_ratio = sum([
    ratio_penalty('ch4','h2o', rmin_ch4_h2o, rmax_ch4_h2o),
    ratio_penalty('co2','h2o', rmin_co2_h2o, rmax_co2_h2o)
])

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

Schedulers/Defaults
	•	Optim: AdamW (decoupled weight decay), cosine anneal + warmup.
	•	Batch size by memory; gradient accumulation for long sequences.
	•	Early stop on val GLL with patience K; keep top‑k checkpoints by GLL.

⸻

9) Splits, Leakage, and Evaluation
	•	GroupKFold by planet_id. Never mix planets across folds.
	•	Val metrics: mean GLL, per‑molecule region error, coverage @ 1-\alpha.
	•	Diagnostics: SHAP magnitude hist, spectral smoothness map, FFT power by cluster.
	•	Ablations: turn off symbolic terms, switch fusion types, encoder isolates.

⸻

10) Repository Topology (authoritative)

.
├── spectramind.py                  # Unified Typer CLI (train/predict/diagnose/calibrate/submit/selftest/ablate)
├── configs/                        # Hydra configs
│   ├── config_v50.yaml
│   ├── data/  model/  train/  diag/
├── src/spectramind/
│   ├── data/                       # loaders, calibration kill-chain (ADC, linearity, dark, flat, CDS/extract)
│   ├── models/                     # FGS1 Mamba SSM, AIRS GAT, μ-decoder, σ-head
│   ├── symbolic/                   # symbolic_loss.py, logic engine, profiles/rules
│   ├── calibration/                # temperature scaling, COREL conformal
│   ├── inference/                  # predict_v50.py (μ/σ), packer
│   ├── training/                   # train_v50.py (MAE→contrastive→supervised)
│   ├── diagnostics/                # SHAP, UMAP/t-SNE/FFT, smoothness, HTML report
│   ├── cli/                        # selftest, pipeline_consistency_checker, wrappers
│   └── utils/                      # logging (console+rotating file+JSONL), hash, env/git capture
├── outputs/                        # run artifacts (DVC-tracked where appropriate)
├── dvc.yaml                        # DAG stages: calibrate→features→train→predict→diagnose
├── pyproject.toml / poetry.lock    # pinned env (Poetry)
├── Dockerfile                      # CUDA runtime parity (optional, dev/prod)
├── .github/workflows/ci.yml        # unit + smoke E2E + reproducibility & diagnostics gates
├── v50_debug_log.md                # human audit log (rotating), appended on every CLI call
└── events.jsonl                    # JSONL event stream (machine telemetry)


⸻

11) CLI Design (Typer + Hydra)

11.1 Commands
	•	spectramind --version → prints CLI version + config hash + timestamp; logs to v50_debug_log.md.
	•	spectramind selftest [--deep] → file presence, shapes, registry, config resolvers; writes to events.jsonl.
	•	spectramind calibrate [--dry-run] [--fast-kaggle] → run kill‑chain; persist calibrated tensors.
	•	spectramind train [--phase mae|contrastive|supervised] → full curriculum with logging/MLflow.
	•	spectramind calibrate-temp → temperature scaling on val.
	•	spectramind calibrate-corel → COREL conformalization, coverage plots.
	•	spectramind predict → μ/σ, pack submission, validate schema.
	•	spectramind diagnose dashboard [--open] → HTML bundle with SHAP/UMAP/FFT/symbolic overlays.
	•	spectramind ablate [--top-n K --md --open-html] → auto ablations with leaderboard export.
	•	spectramind analyze-log [--clean] → parse v50_debug_log.md to Markdown/CSV + heatmaps.
	•	spectramind check-cli-map → command→file mapping, rendered for docs/dashboard.
	•	spectramind corel-train → train COREL GNN/NN conformal model (if used).

11.2 Logging
	•	Console + Rotating file (logs/spectramind.log) + JSONL (events.jsonl).
	•	Each CLI call stamps: ts, git_sha, config_hash, hostname, seeds, cmdline, durations.

⸻

12) Configs & Hydra Composition
	•	configs/config_v50.yaml as root; child trees: data/, model/, train/, diag/.
	•	Key knobs:
	•	model.fgs1.{d_model,n_layers,bidir,dropout}
	•	model.airs.{gat_layers,heads,edge_features}
	•	model.fusion.{type,dim}
	•	loss.{lambda_sm,lambda_nn,lambda_coh,lambda_seam,lambda_ratio,lambda_qm}
	•	train.{optimizer,lr,cosine,warmup,amp,grad_accum,ckpt_n}
	•	calib.{temp.enabled,corel.enabled,alpha}
	•	diag.{umap,tsne,fft,smoothness,html.version}
	•	Seeding: global.seed → torch/cuda/numpy/python; record in logs.

⸻

13) Data & Feature Engineering
	•	FGS1 white‑light: robust aperture photometry; centroid/jitter features; in/out‑of‑transit marks.
	•	AIRS per‑bin: optimal extraction; per‑bin flux time series; seam mask.
	•	Time‑stat features: per‑bin moments (mean, var, skew), FFT power in bands.
	•	Caching: feature hashes; DVC store keyed by planet_id + stage.

⸻

14) Diagnostics & Explainability
	•	SHAP: feature attributions for μ per bin; export bar/hist; overlay symbolic violations.
	•	UMAP/t‑SNE/PCA: latent maps; color by rule violations, entropy, FFT cluster, profile class.
	•	FFT/smoothness maps: per‑bin second‑derivative heatmaps, power spectra clusters.
	•	Symbolic dashboards: top violated rules per planet; ∂L/∂μ symbolic influence maps.
	•	HTML report: versioned diagnostics_v{n}.html, self‑contained; links to artifacts, logs, config snapshot.

⸻

15) Reproducibility Discipline
	•	Code: Git SHA pinned in every artifact.
	•	Config: hash of resolved Hydra config; embed into outputs and logs.
	•	Env: Poetry lock (or Docker image digest).
	•	Data/Models: DVC tracked; stage checksums; manifest JSON in submission.
	•	Telemetry: events.jsonl stream with run graph + durations + resources.

⸻

16) CI/CD & Gates
	•	Jobs: unit → style/lint → smoke E2E (calibrate→train small→predict) → diagnostics export → artifact upload.
	•	Gates: fail if
	•	coverage < target,
	•	val GLL regresses beyond threshold,
	•	calibration coverage broken,
	•	artifacts missing (HTML, logs, schema).
	•	Pages (optional): publish diagnostics HTML + docs site.

⸻

17) Kaggle Runtime Strategy
	•	≤9 h envelope: precompute heavy steps, cache aggressively, checkpoint.
	•	--fast-kaggle: smaller model depth/width, fewer epochs, reduced diagnostics; guaranteed ≤ walltime.
	•	Batch predict: stream planets, low‑mem mode; avoid giant tensors; write partial CSV with flush.
	•	Fallbacks: if COREL too slow, keep temp‑scaled σ; if SHAP heavy, switch to sampling mode.

⸻

18) Security, Safety, and Compliance
	•	No secrets in repo; use env/CI secrets for tokens.
	•	Data licenses: honor challenge license; no redistribution of private sets.
	•	Determinism: guard nondeterministic CUDA kernels when needed; document toggles.
	•	Policy hooks: optional conftest/OPA for deploy artifacts (if serving is enabled).

⸻

19) Risks & Mitigations
	•	σ miscalibration → embrace calibration stage; add z‑score histograms per region.
	•	Seam artifacts → explicit seam continuity term; seam‑aware extraction.
	•	Over‑smoothing → per‑region λ; allow sharp features inside molecule windows.
	•	Runtime overruns → fast mode, staged caching, CI time budgets.
	•	OOD drift → symbolic priors, entropy/violation alarms, ablation re‑tuning.

⸻

20) Future Extensions (Post‑MVP)
	•	Cross‑attentive fusion with explicit wavelength queries.
	•	Temporal decoders for σ (flow‑based calibration by time).
	•	Physics‑guided priors learned from synthetic RT/forward models.
	•	Neural logic layer that learns rule weights per planet family.
	•	Active error mining: loop violation hotspots into curriculum reweighting.

⸻

21) Acceptance Tests & “Done Means”
	1.	spectramind --version writes version + config hash to v50_debug_log.md.
	2.	selftest passes (files, shapes, registries, resolvers).
	3.	Calibration kill‑chain runs; negatives preserved; variance propagated; logs emitted.
	4.	Train (small) completes end‑to‑end; symbolic loss components validated by unit tests.
	5.	Temp scaling improves val GLL; COREL meets coverage target; plots saved.
	6.	Predict produces schema‑valid CSV; validator passes.
	7.	Diagnostics dashboard renders with SHAP/UMAP/FFT/symbolic overlays; linked artifacts present.
	8.	CI green on unit + smoke E2E + diagnostics; artifacts uploaded.

⸻

22) File Ownership & Interfaces
	•	src/spectramind/data/* ↔ configs/data/*: I/O contracts for tensors, masks, variance.
	•	src/spectramind/models/* ↔ configs/model/*: encoder/decoder shapes, fusion dims.
	•	src/spectramind/symbolic/* ↔ configs/model/symbolic.yaml: rule weights, windows, ratios.
	•	src/spectramind/calibration/* ↔ configs/calib/*: temp/COREL settings.
	•	src/spectramind/diagnostics/* ↔ configs/diag/*: plots, embeddings, HTML options.
	•	spectramind.py ↔ v50_debug_log.md / events.jsonl: CLI telemetry.

⸻

23) Notation & Conventions
	•	Scalars italic (e.g., g, \tau); vectors bold (\mathbf{\mu}) where helpful; indices 0‑based for bins.
	•	Units: electrons (e⁻), seconds, microns (µm), ppm; specify in logs and axes.
	•	JSON keys: snake_case; CSV headers: mu_i, sigma_i.
	•	Plots: include seed, git sha, config hash in footer.

⸻

24) Quickstart (Operator Runbook)

# 1) Env
poetry install --no-root && poetry run python -m spectramind --version

# 2) Sanity
poetry run python -m spectramind selftest --deep

# 3) Calibrate + features
poetry run python -m spectramind calibrate

# 4) Train (curriculum)
poetry run python -m spectramind train --phase mae
poetry run python -m spectramind train --phase supervised

# 5) Calibrate uncertainty
poetry run python -m spectramind calibrate-temp
poetry run python -m spectramind calibrate-corel

# 6) Predict + validate
poetry run python -m spectramind predict

# 7) Diagnose
poetry run python -m spectramind diagnose dashboard --open


⸻

25) Appendix — Molecule Windows & Templates (example)
	•	H₂O: windows [i0..i1] (NIR @ ~1.4 µm proxy), [i2..i3] (~6.3 µm band).
	•	CO₂: window around 4.3 µm.
	•	CH₄: window around 3.3 µm.
	•	Seam index: s (instrument‑specific; set in trace_meta).
	•	Templates: nonnegative envelopes t^m normalized to unit L2 on their windows.

(Exact indices and Δλ provided in configs/model/molecule_windows.yaml.)

⸻

26) Appendix — Schema: events.jsonl (telemetry)

Each line:

{
  "ts":"2025-08-12T04:20:31Z",
  "sha":"a1b2c3d",
  "cfg":"e52f…",
  "cmd":"train --phase supervised",
  "dur_ms": 183545,
  "host":"ellks-01",
  "seed": 1337,
  "metrics": {"gll_val": 1.732, "coverage@0.9": 0.905}
}


⸻

27) Appendix — Config Hashing
	•	Resolve Hydra → dump full config → SHA‑256 → run_hash_summary_v50.json.
	•	Log hash in CLI output + dashboard; include in submission manifest.

⸻

28) Appendix — Submission Manifest

submission_manifest.json:

{
  "created":"2025-08-12T04:29:02Z",
  "git_sha":"a1b2c3d",
  "config_hash":"e52f…",
  "temp_scale":"1.084",
  "corel_alpha":0.1,
  "tooling":{"poetry":"1.8.4","python":"3.11","torch":"2.4"},
  "artifacts":{"csv":"submission.csv","html":"diagnostics_v5.html"}
}


⸻

29) Appendix — Troubleshooting (Tactics)
	•	Exploding loss: clamp σ_min, reduce LR, enable AMP w/ dynamic loss scaling, inspect NaNs.
	•	Flat μ: verify fusion dims; check FGS1 input scaling; relax λ_sm in molecule windows.
	•	Coverage < target: re‑run temp scaling; increase COREL quantile; inspect z‑score hist.
	•	Runtime: switch --fast-kaggle, cut SHAP samples, disable t‑SNE, lower encoder depth.
	•	Seam steps: verify seam index and mask; increase λ_seam; re‑extract trace.

⸻

30) License & Credits
	•	Built for NeurIPS 2025 Ariel Data Challenge.
	•	Cite mission and challenge materials as required.
	•	Internal code under project license (see LICENSE).

⸻

This document is living. Every change to the pipeline should land with a diff here and a bumped HTML diagnostics version so operators can trust what they run.
