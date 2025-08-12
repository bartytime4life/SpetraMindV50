⸻

ARCHITECTURE.md

SpectraMind V50 — NASA‑grade Software + Science Architecture for the NeurIPS 2025 Ariel Data Challenge

This document is the single source of truth for SpectraMind V50.
It integrates instrumentation physics, data processing, ML architecture, uncertainty, symbolic constraints, MLOps, and operations into a rigorous, reproducible, review‑ready specification.
If this document and the code ever disagree, the code is wrong.

⸻

0. Mission Directive

Objective. Predict per‑planet transmission spectra as mean (μ) and uncertainty (σ) across 283 wavelength bins, maximizing log‑likelihood under strict runtime limits, with scientific fidelity and explainability.

Commitments.
	•	Physics‑faithful outputs: μ ≥ 0, smooth where physics demands, coherent inside molecule windows, seam‑continuous.
	•	Uncertainty you can trust: σ calibrated post‑hoc (temperature scaling, COREL conformal).
	•	Reproducible to audit standards: code SHA, config hash, env, data/model versions.
	•	Runtime discipline: ≤ 9 hours end‑to‑end; predictable “fast‑kaggle” profile.

⸻

1. Scientific & Competitive Landscape

1.1 Transit Spectroscopy Basics
	•	Signal: During transit, wavelength‑dependent stellar flux drop encodes atmospheric absorption.
	•	Scale height: H=\frac{kT}{\mu g}. Hotter, low‑g atmospheres (large H) imprint deeper spectral features.
	•	Continuum vs lines: Broad “continuum” regions must be smooth; molecule bands (H₂O, CO₂, CH₄) admit sharper structure.

1.2 Instrument & Noise
	•	FGS1 (context channel): Encodes pointing jitter/centroid drift correlated with throughput variations.
	•	AIRS-CH0 (spectral): 283 bins; detector seam introduces potential discontinuity.
	•	Noise/systematics: Stellar activity, intrapixel response, ramps, non‑linearity; handled by calibration + features.

1.3 Challenge Constraints
	•	Metric: Gaussian Log‑Likelihood (GLL) over joint (μ, σ).
	•	Weighting reality: FGS1‑informed modeling is decisive; AIRS bins are individually weaker without context.
	•	OOD: Expect atmospheres beyond training distribution; symbolic physics guards against nonsense.

⸻

2. Level‑1 System Overview

flowchart LR
  R[Raw Cubes: FGS1, AIRS-CH0] --> K[Calibration Kill Chain]
  K --> F[Feature Cache]
  F --> E1[FGS1 Encoder (Mamba SSM)]
  F --> E2[AIRS Encoder (GAT over 283 bins)]
  E1 --> Z[Fusion]
  E2 --> Z
  Z --> D1[μ Decoder (Multi-Scale)]
  Z --> D2[σ Head (Flow/Quantiles)]
  D1 --> S[Symbolic Physics Engine]
  D2 --> S
  S --> C[Uncertainty Calibration (Temp + COREL)]
  C --> P[submission.csv + manifest.json]
  P --> H[Diagnostics HTML, Logs, Telemetry]


⸻

3. Data Contracts

3.1 Directory Topology

/data
  raw/{fgs1,airs_ch0}/planet_*/...
  calibrated/{fgs1,airs}/planet_*.npz
  features/{fgs1_white,airs_bins}/planet_*.npz
  splits/groupkfold.json

3.2 Calibrated Tensors
	•	fgs1_{planet}.npz → frames[T,H,W] : float32, variance[T,H,W] : float32, mask[T,H,W] : bool
	•	airs_{planet}.npz → same keys + trace_meta.json (seam index, dispersion poly, slit geometry)

3.3 Feature Tensors
	•	fgs1_white_{planet}.npz → flux[T], time[T], centroid[T,2], jitter[T,2]
	•	airs_bins_{planet}.npz → flux[T,283], time[T]

3.4 Submission Schema (schemas/submission.schema.json)
	•	Columns: mu_0..mu_282, sigma_0..sigma_282 as decimal strings parseable to float.
	•	Validator must pass before packaging.

⸻

4. Calibration Kill Chain (Level‑2)

Invariants: no negative clipping; propagate variance; log params & deltas at each stage (JSON); DVC‑track every artifact.

	1.	ADC reversal & bias
E=(R-o)\cdot g, \mathrm{Var}_E=g^2\mathrm{Var}_R. Log: gain/bias stats, pre/post histograms.
	2.	Bad pixel mask & interpolation
Mask hot/cold/sat/NaN/persistence/cosmic; interpolate with added floor \epsilon^2. Log: mask fraction, kernel.
	3.	Non‑linearity correction
Per‑pixel f: E_{\text{lin}}=f(E), variance via Jacobian (f’(E))^2\mathrm{Var}_E. Log: median ppm correction.
	4.	Dark subtraction
E_d=E_{\text{lin}}-D, \mathrm{Var}d=\mathrm{Var}{\text{lin}}+\mathrm{Var}_D. Log: dark temperature proxies.
	5.	Flat‑field
E_f=E_d/F, \mathrm{Var}_f=\mathrm{Var}_d/F^2. Log: flat stats, cross‑hatch flags.
	6.	Trace extraction / photometry
FGS1: aperture+annulus; AIRS: optimal extraction along slit; seam mask \mathcal{S}. Log: throughput, seam indices.

Physics rationale. Each correction linearizes or normalizes detector response; propagation preserves uncertainty credibility.

⸻

5. Feature Engineering
	•	FGS1 white‑light: total flux + centroid/jitter; in/out‑of‑transit flags for contrastive stage.
	•	AIRS per‑bin: time‑series per bin; FFT band powers; per‑bin variance & SNR; seam flags.
	•	Caching: feature hashes keyed by planet_id + calibration hash; immutable after write.

⸻

6. Modeling Architecture (Level‑2)

6.1 FGS1 Encoder — Mamba SSM
	•	Input: [flux, centroid_x, centroid_y, jitter_x, jitter_y, …]_{t=1..T}.
	•	O(L) time/memory; AMP‑safe; TorchScript‑ready.
	•	Output latent h_{\text{fgs1}}\in\mathbb{R}^D + optional stepwise latents.

6.2 AIRS Encoder — Physics‑Informed GAT
	•	Graph G=(V,E), |V|=283.
	•	Edges: wavelength adjacency, molecule‑window co‑membership, seam adjacency.
	•	Edge features: \Delta\lambda, molecule tags, seam flag; attention weights exported.

6.3 Fusion & Decoders
	•	Fusion: concat+MLP (default) | cross‑attention | gated; dimension checks at build time.
	•	μ Decoder: multi‑scale (coarse→mid→fine) with skip fusion; auxiliary losses optional.
	•	σ Head: Softplus + \sigma_{\min} or quantile (q10, q50, q90) with monotonic penalty.

⸻

7. Objective Functions

7.1 Gaussian Log‑Likelihood (primary)

\mathcal{L}{\text{gll}}=\frac{1}{2}\sum{i=0}^{282}\Big[\log(2\pi\sigma_i^2)+\frac{(y_i-\mu_i)^2}{\sigma_i^2}\Big]
Use \log\sigma parameterization or clamp \sigma\ge \sigma_{\min} for stability.

7.2 Symbolic Physics Pack (differentiable)
	•	Smoothness (2nd derivative):
\delta_i=\mu_{i+1}-2\mu_i+\mu_{i-1}, L_{sm}=\sum_{i=1}^{281} w^{sm}_i\delta_i^2.
Rationale: physical spectra are smooth outside sharp molecular lines.
	•	Non‑negativity:
L_{nn}=\sum_i w^{nn}_i \mathrm{ReLU}(-\mu_i) (or squared).
Rationale: transit depth can’t be negative.
	•	Molecular coherence:
For molecule m windows W_m, normalized response s^m_i=\mu_i/\max(\epsilon,\|\mu_{W_m}\|2).
L^m{coh}=\sum_{i\in W_m} w^m_i \mathrm{ReLU}(t^m_i-s^m_i)^2.
Rationale: if molecule appears, its envelope must be coherent across its band.
	•	Seam continuity:
L_{seam}=w_{seam}(\mu_{s^-}-\mu_{s^+})^2.
Rationale: detector seam should not create artificial discontinuity.
	•	Chemistry ratios:
A_m=\sum_{i\in W_m}\mu_i\Delta\lambda_i.
L_{ratio}=\sum_{(a,b)}\mathrm{ReLU}(r^{min}{ab}-\frac{A_a}{A_b+\epsilon})+\mathrm{ReLU}(\frac{A_a}{A_b+\epsilon}-r^{max}{ab}).
Rationale: band area ratios constrained by plausible atmospheric chemistry.
	•	Quantile monotonicity (if predicting quantiles):
L_{qm}=\sum_i [\mathrm{ReLU}(q_{10,i}-q_{50,i})+\mathrm{ReLU}(q_{50,i}-q_{90,i})].

Total: \mathcal{L}=\mathcal{L}{\text{gll}}+\lambda{sm}L_{sm}+\lambda_{nn}L_{nn}+\lambda_{coh}\sum_m L^m_{coh}+\lambda_{seam}L_{seam}+\lambda_{ratio}L_{ratio}+\lambda_{qm}L_{qm}.

⸻

8. Training Curriculum
	1.	MAE pretraining: Masked bins; reconstruct spectra; optional log‑σ pretrain.
	2.	Contrastive (optional): Positive (in‑transit) vs negative (out‑of‑transit) to deconfound systematics; InfoNCE on latents.
	3.	Supervised: GLL + symbolic pack; cosine LR, AMP, grad accumulation; checkpoint top‑k by val GLL.
	4.	Calibration: Fit temperature scaling (global τ), then per‑bin COREL quantiles for target coverage.

⸻

9. Splits & Evaluation
	•	Leakage control: GroupKFold by planet_id (no crossing).
	•	Metrics: mean GLL; per‑molecule region error; coverage @ 1-\alpha.
	•	Ablations: fusion types, symbolic toggles, encoders in isolation.

⸻

10. Inference & Packaging
	•	Batching: stream per‑planet; AMP; clamp σ; write CSV incrementally to avoid large buffers.
	•	Validation: schema check; if invalid → hard failure.
	•	Manifest: submission_manifest.json with git SHA, config hash, env, calibration params, toolchain versions.

⸻

11. Uncertainty Calibration

11.1 Temperature Scaling

Fit \tau>0 to minimize val GLL; re‑scale \sigma^\ast=\tau\sigma. Prefer optimizing \log\tau.

11.2 COREL Spectral Conformal

Compute \epsilon_i=|y_i-\mu_i|/\sigma^\ast_i.
Select per‑bin quantile q_i for coverage 1-\alpha.
Final \sigma’_i=q_i\sigma^\ast_i. Emit corel_q_bin.csv + coverage plots.

⸻

12. Diagnostics & Explainability
	•	SHAP on μ with overlays of symbolic violations.
	•	UMAP/t‑SNE/PCA for latent space; color by molecule region, violation rate, entropy.
	•	FFT/smoothness maps diagnose periodicities & roughness.
	•	HTML dashboard (self‑contained): plots, logs, config snapshot, artifact links.

Image placeholders (to be generated by diagnostics):


⸻

13. CLI Contracts (Typer)

spectramind --version
spectramind selftest [--deep]
spectramind calibrate [--dry-run] [--fast-kaggle]
spectramind train --phase {mae,contrastive,supervised}
spectramind calibrate-temp
spectramind calibrate-corel
spectramind predict --out-csv PATH
spectramind diagnose dashboard [--open]
spectramind ablate --top-n K [--md --open-html]
spectramind analyze-log [--clean]

Exit codes: 0 OK; 2 schema violation; 3 bad config; 4 runtime budget risk; 5 internal error.
Every command appends to events.jsonl (machine) and v50_debug_log.md (human).

⸻

14. Configuration (Hydra)
	•	Root: configs/config_v50.yaml
	•	Trees: data.*, model.{fgs1,airs,fusion}, loss.*, train.*, calib.*, diag.*
	•	Hashing: resolved config → SHA‑256; emit to logs & manifest; keys caches.

⸻

15. Logging & Telemetry
	•	Sinks: console, logs/spectramind.log (rotating), events.jsonl.
	•	Event schema (per line):

{"ts":"2025-08-12T04:20:31Z","sha":"a1b2c3d","cfg":"e52f…","cmd":"train --phase supervised","dur_ms":183545,"host":"ellks-01","seed":1337,"metrics":{"gll_val":1.732,"coverage@0.9":0.905}}

	•	Policy: UTC timestamps; no secrets in logs.

⸻

16. Observability & Profiling
	•	Stage timers; GPU/CPU/memory sampling where available.
	•	PyTorch profiler hooks; CSV/HTML exports.
	•	Autograd anomaly mode toggle for debug.

⸻

17. Performance Engineering
	•	AMP everywhere (guard rails for numerics).
	•	Grad accumulation for long sequences; pinned memory.
	•	Vectorized symbolic losses; avoid Python loops in hot paths.
	•	OOM‑retry shim lowers batch size automatically.

⸻

18. Runtime Discipline (Kaggle Envelope)
	•	Budget: ≤ 9 hours.
	•	--fast-kaggle: reduced depth/width, fewer epochs, slim diagnostics.
	•	Checkpoints & caches survive restarts; resume logic built‑in.
	•	Fallbacks: skip heavy SHAP; keep temp‑scaled σ if COREL too slow.

⸻

19. Security & Compliance
	•	No secrets in repo; CI secrets only.
	•	SBOM on release; pinned dependencies; scan supply chain.
	•	Licensing: obey challenge license; do not redistribute private data.

⸻

20. CI/CD Gates
	•	Stages: lint → mypy → unit → smoke E2E → diagnostics export → artifact upload.
	•	Fail if: schema invalid; coverage below target; val GLL regression > threshold; missing artifacts (CSV, HTML, manifest, logs).

⸻

21. Testing Strategy
	•	Unit: calibration math; variance propagation; losses; encoders/decoders interfaces.
	•	Property: enforce invariants (non‑negativity, seam continuity, quantile monotonicity).
	•	Integration: tiny planet E2E; assert contracts; verify artifacts.
	•	Golden files: small diagnostic HTML; schema‑valid CSV samples.

⸻

22. Error Handling & Resilience
	•	Typed exceptions mapped to exit codes.
	•	Atomic writes via *.tmp → rename.
	•	Retry transient IO with backoff; idempotent stages.
	•	Error messages include operator action + doc anchor.

⸻

23. Reproducibility Discipline
	•	Seed torch/cuda/numpy/python from global.seed; log it.
	•	Deterministic flags (optional); document perf tradeoff.
	•	Manifests: git SHA, config hash, toolchain versions, calibration params.
	•	DVC pinning for data and models.

⸻

24. Data Management & Splits
	•	GroupKFold by planet_id; store splits/groupkfold.json (seeded).
	•	Metadata includes seam index, Δλ, molecule windows used by symbolic loss.
	•	Loaders assert split integrity.

⸻

25. Symbolic Loss — Detailed Math & Physics

Let \mu\in\mathbb{R}^{283}, wavelength‑ordered; \Delta\lambda_i>0.
	•	Smoothness: physical atmospheric transmission is differentiable except at genuine lines;
L_{sm}=\sum_{i=1}^{281}w^{sm}i (\mu{i+1}-2\mu_i+\mu_{i-1})^2
with higher w^{sm} in continuum regions.
	•	Non‑negativity: transit depth cannot be negative;
L_{nn}=\sum_i w^{nn}_i\mathrm{ReLU}(-\mu_i).
	•	Molecular coherence: within windows W_m, normalize and enforce template envelopes t^m\ge0;
s^m_i=\frac{\mu_i}{\max(\epsilon,\|\mu_{W_m}\|2)},\quad
L^m{coh}=\sum_{i\in W_m} w^m_i\,\mathrm{ReLU}(t^m_i-s^m_i)^2
	•	Seam continuity: avoid instrument‑induced steps;
L_{seam}=w_{seam}(\mu_{s^-}-\mu_{s^+})^2.
	•	Band area ratios: chemistry plausibility;
A_m=\sum_{i\in W_m}\mu_i\Delta\lambda_i,
L_{ratio}=\sum_{(a,b)} \mathrm{ReLU}(r^{min}{ab}-\frac{A_a}{A_b+\epsilon})+\mathrm{ReLU}(\frac{A_a}{A_b+\epsilon}-r^{max}{ab}).
	•	Quantile monotonicity: if predicting (q_{10},q_{50},q_{90}), enforce q_{10}\le q_{50}\le q_{90}.

⸻

26. COREL Conformal — Procedure & Verification
	1.	Train model; apply temperature scaling to σ → \sigma^\ast.
	2.	On validation, compute \epsilon_i=|y_i-\mu_i|/\sigma^\ast_i.
	3.	Choose quantile q_i per bin for target coverage 1-\alpha.
	4.	Predict with \sigma’_i=q_i\sigma^\ast_i.
	5.	Verify coverage; save plots & q_i table; CI gate coverage ≥ target−tol.

⸻

27. Local Development & Workstations
	•	Poetry environment (pyproject.toml) with GPU PyTorch.
	•	Optional CUDA Docker for parity.
	•	Pre‑commit (ruff, mypy, black).
	•	Optional local LLMs (Ollama/HF accelerate) for code/data assistants — off by default.

⸻

28. Release & Versioning
	•	SemVer tags; lock Poetry + Docker digest; attach SBOM.
	•	Release notes from conventional commits; attach diagnostic HTML.
	•	Archive submission manifests for reproducibility.

⸻

29. Risk Register

ID	Risk	Prob.	Impact	Mitigation
R1	σ miscalibration	M	H	Temp scaling + COREL; coverage CI gate
R2	Seam artifact	M	M	Seam penalty; seam‑aware extraction; diagnostics flag
R3	Over‑smooth μ	M	M	Region‑wise λ; exempt molecule cores; ablation checks
R4	Runtime overrun	M	H	--fast-kaggle profile; cache; early‑exit on heavy plots
R5	OOD drift	M	M	Symbolic priors; violation alarms; curriculum reweighting
R6	Data leakage	L	H	GroupKFold by planet; loader assertions; tests
R7	Non‑determinism	L	M	Seed & deterministic flags; log env; CI reproducibility


⸻

30. Acceptance Criteria (Go/No‑Go)
	1.	spectramind --version logs version + config hash.
	2.	selftest passes (files, schemas, registries).
	3.	Calibration runs with variance propagation, no negative clipping, logs written.
	4.	Training completes curriculum; val GLL improves; unit/property tests pass.
	5.	Temp scaling improves val GLL; COREL achieves coverage target with plots.
	6.	Prediction writes schema‑valid CSV + manifest.
	7.	Diagnostics HTML bundles SHAP/UMAP/FFT/symbolic overlays with provenance.
	8.	CI green: lint, mypy, unit, smoke E2E, diagnostics; gates satisfied.
	9.	Total runtime within budget; --fast-kaggle hits guaranteed envelope.

⸻

31. Glossary
	•	FGS1 — Fine Guidance Sensor channel (context; jitter/centroid).
	•	AIRS — Spectrometer generating 283 bins.
	•	Mamba/SSM — State Space Model for long sequences.
	•	GAT — Graph Attention Network.
	•	GLL — Gaussian Log‑Likelihood.
	•	COREL — Conformal calibration per spectral bin.
	•	DVC — Data Version Control.

⸻

32. Quickstart (Operator Runbook)

# Environment
poetry install --no-root && poetry run python -m spectramind --version

# Sanity
poetry run python -m spectramind selftest --deep

# Calibration & features
poetry run python -m spectramind calibrate

# Training
poetry run python -m spectramind train --phase mae
poetry run python -m spectramind train --phase supervised

# Uncertainty calibration
poetry run python -m spectramind calibrate-temp
poetry run python -m spectramind calibrate-corel

# Predict & package
poetry run python -m spectramind predict --out-csv outputs/submission.csv

# Diagnostics
poetry run python -m spectramind diagnose dashboard --open


⸻

33. Reference Pseudocode — Vectorized Loss

def total_loss(mu, sigma, y, valid_mask, seam_idx, windows, templates, dl, lam,
               sigma_min=1e-4, eps=1e-6):
    # Gaussian log-likelihood
    sigma = sigma.clamp_min(sigma_min)
    res = (y - mu) / sigma
    L_gll = 0.5 * (res**2 + 2.0 * sigma.log() + math.log(2*math.pi))
    L_gll = (L_gll * valid_mask).sum(-1).mean()

    # Smoothness
    d2 = mu[..., 2:] - 2*mu[..., 1:-1] + mu[..., :-2]
    w_sm = smoothness_weights(valid_mask[1:-1]).to(mu)
    L_sm = (w_sm * d2**2).sum(-1).mean()

    # Non-negativity
    L_nn = torch.relu(-mu).sum(-1).mean()

    # Molecular coherence
    L_coh = 0.0
    for m, idx in windows.items():
        mu_m = mu.index_select(-1, idx)
        t_m  = templates[m].to(mu)
        norm = mu_m.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        s_m  = mu_m / norm
        L_coh = L_coh + (torch.relu(t_m - s_m)**2).sum(-1).mean()

    # Seam continuity
    s = seam_idx
    L_seam = ((mu[..., s-1] - mu[..., s])**2).mean()

    # Band area ratios
    def band_area(idx):
        return (mu.index_select(-1, idx) * dl.index_select(0, idx)).sum(-1)
    areas = { m: band_area(windows[m]) for m in windows }
    def ratio_penalty(a, b, rmin, rmax):
        r = areas[a] / (areas[b] + eps)
        return (torch.relu(rmin - r) + torch.relu(r - rmax)).mean()
    L_ratio = ratio_penalty('ch4','h2o', *R_CH4_H2O) + ratio_penalty('co2','h2o', *R_CO2_H2O)

    # Quantile monotonicity (optional: q10,q50,q90 tensors)
    L_qm = 0.0

    L_sym = lam.sm*L_sm + lam.nn*L_nn + lam.coh*L_coh + lam.seam*L_seam + lam.ratio*L_ratio + lam.qm*L_qm
    return L_gll + L_sym


⸻

34. Governance
	•	Any change to contracts (schemas, CLI, directory layout, loss math) must update this document in the same PR.
	•	CI tests verify doc/code alignment (hash checks, help text diffs).
	•	Reviewers must confirm compliance with §§1, 3–7, 11, 20, 30.

⸻

35. Visual Appendix Placeholders (to generate)

assets/
  sample_spectra_overlay.png            # μ vs λ with molecule bands annotated
  symbolic_violation_heatmap.png        # planet × rule heatmap
  latent_umap.png                       # encoder latent UMAP coloring by violation
  fft_power.png                         # FFT power bands across bins
  calibration_coverage.png              # coverage before/after temp+COREL
  pipeline_block_diagram.png            # optional block diagram export


⸻

This document controls the build.
Implementations must trace back to the requirements and equations here.
Discipline wins. Science leads. Reproducibility is non‑negotiable.