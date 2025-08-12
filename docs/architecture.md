ARCHITECTURE.md

SpectraMind V50 — NASA‑grade software and systems architecture for the NeurIPS 2025 Ariel Data Challenge

This is the master engineering doctrine for SpectraMind V50. It specifies standards, interfaces, contracts, algorithms, tooling, and operational policy. It is the single source of truth for developers, reviewers, CI, and operations.

⸻

0. Mission Directive
	•	Deliver leaderboard‑winning, physics‑faithful predictions (μ, σ per spectral bin) with calibrated uncertainty under strict runtime limits.
	•	Maintain audit‑grade reproducibility (code SHA, environment, config hash, data versions) for every artifact.
	•	Enforce engineering discipline: deterministic builds, typed code, CLI contracts, schema validation, coverage & quality gates.

⸻

1. Governing Standards & Practices
	•	Languages: Python 3.11 (primary), Bash (ops), YAML (configs), Markdown (docs).
	•	Style: PEP8 + ruff; type hints enforced with mypy (strict optional).
	•	Packaging: Poetry (pyproject.toml) pinned; optional Docker (CUDA runtime parity).
	•	Config: Hydra + structured dataclasses; single root config_v50.yaml.
	•	Data versioning: DVC; artifacts keyed by planet ID and stage hash.
	•	Experiment tracking: JSONL event stream (always), optional MLflow/W&B.
	•	CLI: Typer app (spectramind.py) with stable subcommand API and exit codes.
	•	Logging: console + rotating file + JSONL; no noisy prints; timestamps in UTC.
	•	Security: zero secrets in repo; environment/CI secrets only; SBOM export on release.
	•	Documentation: this file is authoritative; MkDocs site optional for rendering.

⸻

2. Repository Layout (Authoritative)

.
├── spectramind.py                    # Unified Typer CLI
├── configs/                          # Hydra root and trees
│   ├── config_v50.yaml
│   ├── data/*.yaml
│   ├── model/*.yaml
│   ├── train/*.yaml
│   └── diag/*.yaml
├── src/spectramind/
│   ├── data/                         # loaders & calibration kill chain
│   ├── features/                     # feature builders, caching
│   ├── models/                       # encoders/decoders/heads
│   ├── symbolic/                     # differentiable physics constraints
│   ├── calibration/                  # temperature scaling, conformal (COREL)
│   ├── training/                     # loops, schedulers, curriculum
│   ├── inference/                    # batch prediction, packers
│   ├── diagnostics/                  # explainability, dashboards
│   └── utils/                        # logging, schema, hashing, env capture
├── schemas/
│   └── submission.schema.json        # CSV contract
├── data/                             # local cache root (DVC tracked where applicable)
├── outputs/                          # run artifacts (model ckpts, eval, html)
├── dvc.yaml                          # DAG: calibrate→features→train→predict→diagnose
├── pyproject.toml / poetry.lock      # pinned env
├── Dockerfile                        # optional parity container
├── .github/workflows/ci.yml          # CI gates
├── logs/spectramind.log              # rotating file log
├── v50_debug_log.md                  # human audit log per CLI call
└── events.jsonl                      # JSONL telemetry stream


⸻

3. System Overview (L1)

flowchart LR
  A[Raw Telemetry: FGS1, AIRS-CH0] --> B[Calibration Kill Chain]
  B --> C[Feature Cache]
  C --> D1[FGS1 Encoder (SSM/Mamba)]
  C --> D2[AIRS Encoder (GAT)]
  D1 --> E[Fusion]
  D2 --> E
  E --> F1[μ Decoder (Multi-scale)]
  E --> F2[σ Head (Flow/Quantiles)]
  F1 --> G[Symbolic Physics Engine]
  F2 --> G
  G --> H[Uncertainty Calibration (Temp + COREL)]
  H --> I[CSV Submission + Manifest]
  I --> J[Diagnostics & HTML]


⸻

4. Data Contracts

4.1 Directory Topology

/data
  raw/{fgs1,airs_ch0}/planet_*/...
  calibrated/{fgs1,airs}/planet_*.npz
  features/{fgs1_white,airs_bins}/planet_*.npz
  splits/groupkfold.json

4.2 Calibrated Tensors
	•	fgs1_{planet}.npz:
	•	frames: float32[T,H,W]
	•	variance: float32[T,H,W] (propagated)
	•	mask: bool[T,H,W]
	•	airs_{planet}.npz: same keys + trace_meta: json (seam, dispersion, slit geometry)

4.3 Feature Tensors
	•	fgs1_white_{planet}.npz: flux[T], time[T], centroid[T,2], jitter[T,2]
	•	airs_bins_{planet}.npz: flux[T,283], time[T]

4.4 Submission Schema
	•	CSV columns: mu_0..mu_282, sigma_0..sigma_282
	•	schemas/submission.schema.json is the contract; schema_check.py validates before submit.

⸻

5. Calibration Kill Chain (L2)

Invariants: no negative clipping; explicit variance propagation; per‑stage JSON log (duration, params, deltas).

	1.	ADC reversal & bias
E=(R-o)\cdot g, \mathrm{Var}_E = g^2 \mathrm{Var}_R
Log g, o, pre/post stats.
	2.	Bad pixel mask & interpolation
Hot/cold/NaN/sat/persistence/cosmic → masked; interpolated with noise floor \epsilon^2.
Log mask fraction, kernel.
	3.	Non‑linearity correction
Per‑pixel LUT/poly f: E_{\text{lin}} = f(E), variance via f’(E).
Log median correction (ppm), clips.
	4.	Dark subtraction
E_d = E_{\text{lin}}-D, \mathrm{Var}d=\mathrm{Var}{\text{lin}}+\mathrm{Var}_D.
Log dark temperature proxy if present.
	5.	Flat‑field
E_f = E_d/F, \mathrm{Var}_f=\mathrm{Var}_d/F^2.
Log flat stats; cross‑hatch flags.
	6.	Trace extraction / photometry
	•	FGS1 white‑light: aperture+annulus; variance sums.
	•	AIRS spectral: optimal extraction along slit; seam mask \mathcal{S}.
Log throughput, seam indices.

All steps DVC‑tracked; caches keyed by planet id + stage hash.

⸻

6. Feature Engineering
	•	FGS1: flux, centroid, jitter, in/out‑of‑transit marks; detrend candidates.
	•	AIRS: per‑bin time series; FFT bands, moments; seam flags as features.
	•	Caching: hashed on inputs + params; immutable once written; provenance stored.

⸻

7. Modeling Architecture (L2)

7.1 FGS1 Encoder (SSM/Mamba)
	•	Inputs: [flux, centroid_x, centroid_y, jitter_x, jitter_y, ...] over time.
	•	Bi‑directional SSM; O(L) memory/time; AMP‑safe; TorchScript‑ready.
	•	Outputs: pooled latent h_{\text{fgs1}}\in\mathbb{R}^D; optional per‑step latents for UMAP/SHAP.

7.2 AIRS Encoder (Physics‑informed GAT)
	•	Graph G=(V,E), |V|=283.
	•	Edges: wavelength adjacency, molecular co‑windows, seam continuity links.
	•	Edge features: \Delta\lambda, molecule tags, seam flag.
	•	Returns h_{\text{airs}}\in\mathbb{R}^D + attention maps for diagnostics.

7.3 Fusion
	•	concat+MLP (default), cross-attend, or gated.
	•	Shape checks enforced at build time; Hydra config defines type/dims.

7.4 Decoders
	•	μ Decoder (multi‑scale): coarse→mid→fine; skip connections; auxiliary losses optional.
	•	σ Head (flow/quantiles): Softplus + \sigma_{\min}; or quantiles (q10,q50,q90) with monotonic penalties.

⸻

8. Objectives

8.1 Gaussian Log‑Likelihood (primary)

\mathcal{L}{\text{gll}}=\frac{1}{2}\sum_i\big[\log(2\pi\sigma_i^2)+\frac{(y_i-\mu_i)^2}{\sigma_i^2}\big]
Clamp \sigma \ge \sigma{\min}; prefer log_sigma param for stability.

8.2 Symbolic Physics Pack (differentiable)
	•	Smoothness \partial^2\mu: central difference L2; region‑weighted.
	•	Non‑negativity: \mathrm{ReLU}(-\mu) or squared.
	•	Molecular coherence: normalized response vs non‑negative template envelopes t^m.
	•	Seam continuity: penalty at seam index s.
	•	Chemistry ratios: CH₄:H₂O:CO₂ band area envelopes.
	•	Quantile monotonicity: enforce q_{10}\le q_{50}\le q_{90}.

Total loss: \mathcal{L}=\mathcal{L}{\text{gll}}+\mathcal{L}{\text{sym}}.

⸻

9. Training Curriculum
	1.	MAE pretrain: masked spectral bins; optional σ pretrain (predict log‑σ).
	2.	Contrastive (optional): in‑transit vs out‑of‑transit; InfoNCE on latents.
	3.	Supervised: GLL + symbolic; cosine LR, AMP, grad accumulation; early stop on val GLL.
	4.	Calibration: temperature scaling → COREL; save q_i per bin.

Schedulers: AdamW + cosine; seed everything; checkpoint top‑k by val GLL.

⸻

10. Splits & Evaluation
	•	GroupKFold by planet_id (no leakage).
	•	Metrics: mean GLL, molecule‑region error, coverage @ 1-\alpha.
	•	Ablations: fusion type, symbolic toggles, encoder isolates; tracked in JSONL.

⸻

11. Inference & Submission
	•	Stream planets; low‑mem batches; AMP; write partial CSV with flush.
	•	Validate CSV against submission.schema.json (fail fast).
	•	Produce submission_manifest.json with run metadata: time, SHA, config hash, calib params, toolchain versions.

⸻

12. Uncertainty Calibration
	•	Temperature scaling: learn global \tau on validation; \sigma^\ast=\tau\sigma.
	•	COREL spectral conformal: z‑score quantiles per bin → per‑bin q_i, calibrated \sigma’_i=q_i\sigma^\ast_i.
	•	Coverage targets recorded with histograms; failure gates in CI.

⸻

13. Diagnostics & Explainability
	•	SHAP attributions for μ; overlay symbolic violation heatmaps.
	•	UMAP/t‑SNE/PCA of latents; color by violation, entropy, FFT class.
	•	Smoothness/FFT maps per bin; molecule window overlays.
	•	HTML dashboard bundle (self‑contained), linked from run manifest.

⸻

14. CLI Contracts (Typer)

spectramind --version                # prints version, config hash; logs to v50_debug_log.md
spectramind selftest [--deep]        # repository & config sanity, schemas, registry
spectramind calibrate [--dry-run]    # run kill chain; write calibrated tensors
spectramind train --phase {mae,supervised,contrastive}
spectramind calibrate-temp           # temperature scaling
spectramind calibrate-corel          # conformal per-bin scaling
spectramind predict --out-csv PATH   # batch μ/σ; validate schema; write manifest
spectramind diagnose dashboard       # HTML diagnostics bundle
spectramind ablate --top-n K         # automatic ablations & report
spectramind analyze-log              # parse logs → md/csv

	•	Exit codes: 0 success; 2 schema violation; 3 config error; 4 runtime limit risk; 5 internal error.
	•	Every command appends to events.jsonl and v50_debug_log.md.

⸻

15. Configuration (Hydra)
	•	Root: configs/config_v50.yaml
	•	Key trees:
	•	data.* (paths, io, cache policy)
	•	model.fgs1.*, model.airs.*, model.fusion.*
	•	loss.* (λ weights)
	•	train.* (optimizer, lr, amp, grad_accum)
	•	calib.* (temp, corel)
	•	diag.* (plots, html)
	•	Config hashing: resolve → SHA‑256 → emit in logs + manifest; used for cache keys.

⸻

16. Logging & Telemetry
	•	Console (human) + rotating file (logs/spectramind.log) + JSONL (events.jsonl).
	•	Event schema (per line):

{"ts":"2025-08-12T04:20:31Z","sha":"a1b2c3d","cfg":"e52f…","cmd":"train --phase supervised",
 "dur_ms":183545,"host":"ellks-01","seed":1337,
 "metrics":{"gll_val":1.732,"coverage@0.9":0.905}}

	•	Never print secrets; always UTC timestamps.

⸻

17. Observability & Profiling
	•	Built‑in timers per stage; GPU/CPU/memory sampling (if available).
	•	Autograd anomaly mode togglable for debug.
	•	PyTorch profiler hooks for hot paths; CSV summaries archived.

⸻

18. Performance Engineering
	•	AMP everywhere unless disabled by guardrail.
	•	Gradient accumulation for long sequences; pinned memory dataloaders.
	•	Scalable batch sizing with OOM‑retry shim.
	•	Vectorized symbolic losses; avoid Python loops on critical paths.

⸻

19. Runtime Discipline (Kaggle Envelope)
	•	Walltime budget: ≤ 9 hours total.
	•	--fast-kaggle profile: reduced depth/width, fewer epochs, slim diagnostics; guaranteed time.
	•	Checkpoints & cache to survive kernel restarts; resume logic built‑in.
	•	Fallback plans: if COREL slow, keep temp‑scaled σ; if SHAP heavy, sample mode.

⸻

20. Security & Compliance
	•	No credentials in code/config; use env/CI secrets.
	•	SBOM on release; pinned dependency versions; supply chain scanning.
	•	Dataset license compliance; no re‑distribution of private data.
	•	Optional OPA/conftest policy checks on artifacts before submit.

⸻

21. CI/CD Gates
	•	Stages: lint → mypy → unit → smoke E2E (calibrate→train small→predict) → diagnostics export.
	•	Fail gates:
	•	Submission schema invalid
	•	Coverage (conformal) < target
	•	Val GLL regression > threshold
	•	Required artifacts missing (CSV, HTML, manifest, logs)

⸻

22. Testing Strategy
	•	Unit: calibration transforms, variance propagation math, loss components, encoders/decoders IO.
	•	Property tests: symbolic term invariants (e.g., non‑negativity, seam continuity).
	•	Integration: mini‑planet pipeline; validates end‑to‑end contracts.
	•	Golden files: schema outputs and small diagnostics snapshots.

⸻

23. Error Handling & Resilience
	•	Raise typed exceptions; convert to CLI exit codes.
	•	Partial writes guarded by temp files → atomic rename.
	•	Retries for transient IO; exponential backoff; idempotent stages.
	•	Clear messaging for operator action; link to docs section in error text.

⸻

24. Reproducibility Discipline
	•	Seeds: torch/cuda/numpy/python set via global.seed.
	•	Determinism: optional CUDA deterministic flags; document perf tradeoffs.
	•	Manifests: include git SHA, config hash, environment fingerprints, calibration parms.
	•	DVC: pin data/model versions used in the run; upload on CI if configured.

⸻

25. Data Management & Splits
	•	GroupKFold by planet_id; store splits/groupkfold.json with seeds and rationale.
	•	Enforce no leakage at loader; assert fold constraints in tests.
	•	Augment metadata with seam indices, molecule windows, Δλ per bin for symbolic.

⸻

26. Symbolic Loss — Detailed Math

Let \mu\in\mathbb{R}^{283} wavelength ordered.
	•	Smoothness
\delta_i=\mu_{i+1}-2\mu_i+\mu_{i-1},
L_{sm}=\sum_{i=1}^{281}w^{sm}_i\delta_i^2.
	•	Non‑negativity
L_{nn}=\sum_i w^{nn}_i \mathrm{ReLU}(-\mu_i) (or squared variant).
	•	Molecular coherence
For molecule m windows W_m, template t^m\ge0:
s^m_i=\mu_i/\max(\epsilon,\|\mu_{W_m}\|2),
L^m{coh}=\sum_{i\in W_m}w^m_i\mathrm{ReLU}(t^m_i-s^m_i)^2.
	•	Seam continuity
L_{seam}=w_{seam}(\mu_{s^-}-\mu_{s^+})^2.
	•	Band area ratios
A_m=\sum_{i\in W_m}\mu_i\Delta\lambda_i;
L_{ratio}=\sum_{(a,b)} \mathrm{ReLU}(r^{min}{ab}-\frac{A_a}{A_b+\epsilon})+\mathrm{ReLU}(\frac{A_a}{A_b+\epsilon}-r^{max}{ab}).
	•	Quantile monotonicity
L_{qm}=\sum_i[\mathrm{ReLU}(q_{10,i}-q_{50,i})+\mathrm{ReLU}(q_{50,i}-q_{90,i})].
	•	Total
L_{sym}=\lambda_{sm}L_{sm}+\lambda_{nn}L_{nn}+\lambda_{coh}\sum_m L^m_{coh}+\lambda_{seam}L_{seam}+\lambda_{ratio}L_{ratio}+\lambda_{qm}L_{qm}.

⸻

27. COREL Conformal — Procedure
	1.	Compute validation z‑scores \epsilon_i=|y_i-\mu_i|/\sigma^\ast_i.
	2.	For each bin i, select quantile q_i to target coverage 1-\alpha.
	3.	Calibrated \sigma’_i=q_i\sigma^\ast_i.
	4.	Emit corel_q_bin.csv, coverage plots, z‑score histograms.
	5.	CI gate on coverage: fail if < target by tolerance.

⸻

28. Local Development & Workstations
	•	Poetry environment with GPU PyTorch build.
	•	Optional Docker for CUDA parity; mount project and data.
	•	Local LLMs (Ollama / HF accelerate) for explain/debug (optional hooks).
	•	Dev helpers: pre‑commit (ruff, mypy, black), make dev bootstrap.

⸻

29. Release & Versioning
	•	Tag: vMAJOR.MINOR.PATCH → freeze Poetry lock, Docker image digest, SBOM.
	•	Release notes auto‑generated from conventional commits; attach diagnostics HTML sample.
	•	Archive manifests for reproducibility.

⸻

30. Risk Register & Mitigations
	•	σ miscalibration → enforce temp scaling + COREL; monitor coverage hist.
	•	Seam artifacts → seam penalty + seam‑aware extraction; alerts in diagnostics.
	•	Over‑smooth μ → region‑wise λ; exemptions inside molecule windows.
	•	Runtime overruns → fast profile, ablation toggles, early‑exit safeguards.
	•	OOD drift → symbolic priors + violation alarms; curriculum reweighting.

⸻

31. Acceptance Criteria (Go/No‑Go)
	1.	selftest passes; schemas and registries ok.
	2.	Calibrate→features runs with variance propagation and no negative clipping.
	3.	Train completes curriculum; unit & property tests pass.
	4.	Temp scaling improves val GLL; COREL meets coverage target.
	5.	Predict creates schema‑valid CSV; validator ok; manifest present.
	6.	Diagnostics HTML renders with overlays; artifacts linked.
	7.	CI green: lint, mypy, unit, smoke E2E, diagnostics; gates satisfied.
	8.	Walltime within Kaggle budget with --fast-kaggle.

⸻

32. Glossary
	•	FGS1: Fine Guidance Sensor channel (context/jitter proxy).
	•	AIRS‑CH0: Spectral channel producing 283 bins for μ/σ prediction.
	•	SSM/Mamba: State Space Model family used for efficient sequence encoding.
	•	GAT: Graph Attention Network over wavelength bins.
	•	COREL: Conformal calibration of residuals at spectral resolution.
	•	GLL: Gaussian Log‑Likelihood metric.
	•	DVC: Data Version Control.

⸻

33. Quickstart (Operator Runbook)

# Environment
poetry install --no-root
poetry run python -m spectramind --version

# Sanity
poetry run python -m spectramind selftest --deep

# Calibration & features
poetry run python -m spectramind calibrate

# Training
poetry run python -m spectramind train --phase mae
poetry run python -m spectramind train --phase supervised

# Calibration of uncertainty
poetry run python -m spectramind calibrate-temp
poetry run python -m spectramind calibrate-corel

# Inference & packaging
poetry run python -m spectramind predict --out-csv outputs/submission.csv

# Diagnostics
poetry run python -m spectramind diagnose dashboard


⸻

34. Reference Pseudocode — Loss Pack (Vectorized)

def loss_total(mu, sigma, y, valid, seam_idx, win_map, templates, dl,
               lmbd, sigma_min=1e-4, eps=1e-6):
    # GLL
    sigma = sigma.clamp_min(sigma_min)
    res = (y - mu) / sigma
    L_gll = 0.5 * ((res**2) + 2.0 * sigma.log() + math.log(2*math.pi))
    L_gll = (L_gll * valid).sum(-1).mean()

    # Smoothness
    d2 = mu[..., 2:] - 2*mu[..., 1:-1] + mu[..., :-2]
    w_sm = smoothness_weights(valid[1:-1]).to(mu)
    L_sm = (w_sm * d2**2).sum(-1).mean()

    # Non-negativity
    L_nn = torch.relu(-mu).sum(-1).mean()

    # Molecular coherence
    L_coh = 0.0
    for m, idx in win_map.items():
        mu_m = mu.index_select(-1, idx)
        t_m  = templates[m].to(mu)
        norm = mu_m.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        s_m  = mu_m / norm
        L_coh = L_coh + (torch.relu(t_m - s_m)**2).sum(-1).mean()

    # Seam continuity
    s = seam_idx
    L_seam = ((mu[..., s-1] - mu[..., s])**2).mean()

    # Ratio constraints
    def band_area(idx):
        return (mu.index_select(-1, idx) * dl.index_select(0, idx)).sum(-1)
    areas = { m: band_area(win_map[m]) for m in win_map }
    def ratio_penalty(a, b, rmin, rmax):
        r = areas[a] / (areas[b] + eps)
        return (torch.relu(rmin - r) + torch.relu(r - rmax)).mean()
    L_ratio = ratio_penalty('ch4','h2o', *R_CH4_H2O) + ratio_penalty('co2','h2o', *R_CO2_H2O)

    # Quantile monotonicity (optional)
    L_qm = 0.0  # if q10/q50/q90 present, enforce monotonicity

    L_sym = (lmbd.sm*L_sm + lmbd.nn*L_nn + lmbd.coh*L_coh +
             lmbd.seam*L_seam + lmbd.ratio*L_ratio + lmbd.qm*L_qm)
    return L_gll + L_sym


⸻

35. Governance
	•	Any change to contracts (schemas, CLI args, directory layout, objective math) must update this file in the same PR.
	•	CI enforces doc drift detection (hash of ARCHITECTURE.md referenced in tests).
	•	Reviewers must sign off on compliance with sections 1 (Standards), 4 (Contracts), 21 (CI Gates), and 31 (Acceptance).

⸻

This document controls the build.
If code and this document disagree, the code is wrong. Update the code or update this document in the same change.