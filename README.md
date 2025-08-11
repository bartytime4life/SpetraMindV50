⸻


# SpectraMind V50 — Engineering Vision & Master Architecture
**Neuro-Symbolic, Physics-Informed AI Pipeline for the NeurIPS 2025 Ariel Data Challenge**

> **North Star**: Deliver a reproducible, explainable, physics-informed system that ingests Ariel FGS1/AIRS cubes, outputs μ and σ for 283 bins, passes calibration and diagnostics, and packages a competition-valid submission — all within Kaggle’s 9-hour runtime envelope [oai_citation:1‡Repo and project structure start (North Star).pdf](file-service://file-48QwRwyruT8r78LkzfvJC3).

---

## 0) Purpose & Scope

This document is the **engineering blueprint** for SpectraMind V50:
- **For engineers**: defines modules, contracts, workflows, performance budgets, and acceptance criteria.
- **For scientists**: encodes the physics and symbolic priors into the machine learning architecture.
- **For ops/MLOps**: prescribes the reproducibility, CI/CD, and data governance practices.

It is **authoritative** — deviations must be documented in `docs/adr/` and reviewed.

---

## 1) Scientific & Strategic Drivers

**Challenge**: Predict μ (mean transmission spectrum) and σ (uncertainty) for each bin given FGS1/AIRS time series [oai_citation:2‡NeurIPS Ariel Challenge AI Guide_.txt](file-service://file-M7eFFf3BpjzCaFXKH9513K).

**Key facts**:
- **FGS1-first strategy**: ~58× weight vs any AIRS bin; drives GLL score [oai_citation:3‡NeurIPS Ariel Challenge AI Guide_.txt](file-service://file-M7eFFf3BpjzCaFXKH9513K).
- **Noise model**: astrophysical (limb darkening, stellar activity) + instrumental (jitter, non-linearity, 1/f noise).
- **Metric**: Gaussian Log-Likelihood — rewards calibrated σ, penalizes over/underconfidence.
- **OOD**: unique stellar models per planet — symbolic constraints act as distribution-invariant priors.
- **Runtime**: ≤9h total on Kaggle for ~1,100 planets ⇒ <30s/planet.

---

## 2) Architecture Overview

**System Diagram**:

[FGS1 raw] –┐
│–> [Calibration Kill Chain] –> [FGS1MambaEncoder] –┐
[AIRS raw] –┘                                                       │
–> [Fusion] –> [μ Decoder] –> μ
–> [Fusion] –> [σ Head]   –> σ
[Symbolic Loss Engine]
[Diagnostics & HTML Report]

**Core Components**:
- **Encoders**:
  - `FGS1MambaEncoder` — bidirectional Mamba SSM, TorchScript-safe [oai_citation:4‡Repo and project structure start (North Star).pdf](file-service://file-48QwRwyruT8r78LkzfvJC3).
  - `AIRSSpectralGNN` — GAT graph with edges for wavelength adjacency, molecular bands, detector seams.
- **Fusion**: injects FGS1 latent context into AIRS graph or global token.
- **Decoders**:
  - `MultiScaleDecoder` — coarse→fine μ reconstruction with skip fusion.
  - `FlowUncertaintyHead` — heteroscedastic Softplus σ; optional quantile head with monotonicity loss.
- **Symbolic Loss Engine**:
  - Smoothness (∂²μ penalty), non-negativity, molecular coherence masks, seam continuity, CH₄:H₂O:CO₂ envelopes.
- **Calibration**:
  - Temp scaling + COREL binwise conformalization with molecule-region coverage plots.
- **Diagnostics**:
  - SHAP × symbolic overlays, UMAP/t-SNE, FFT/smoothness maps, symbolic violation tables.

---

## 3) Module Contracts

| Module                                | Input                                      | Output                                | Acceptance Criteria |
|---------------------------------------|--------------------------------------------|----------------------------------------|---------------------|
| `FGS1MambaEncoder`                     | [B, T, C] FGS1 calibrated sequence         | [B, latent_dim]                        | Matches config dims, TorchScript-safe |
| `AIRSSpectralGNN`                      | [B, N, C] AIRS calibrated spectra          | [B, latent_dim]                        | Respects graph schema & edge features |
| `MultiScaleDecoder`                    | latent                                     | μ: [B, 283]                            | Smoothness preserved in coarse bins |
| `FlowUncertaintyHead`                  | latent, μ                                  | σ: [B, 283]                            | Non-negative, calibratable |
| `symbolic_loss.py`                     | μ, σ, metadata                             | scalar loss, per-rule map               | Vectorized, differentiable, per-rule logging |
| `calibrate_temp.py`                    | σ, targets                                 | σ′                                     | Improves GLL on validation |
| `spectral_conformal.py` (COREL)        | μ, σ, targets                              | σ′ per bin                             | Meets coverage targets per region |
| `generate_html_report.py`              | all diagnostics artifacts                  | HTML                                   | Loads in headless browser, embeds provenance |

---

## 4) Pipeline Phases & Acceptance Tests [oai_citation:5‡Repo and project structure start (North Star).pdf](file-service://file-48QwRwyruT8r78LkzfvJC3)

**Phase 0 — Repo Hardening & Run Control**
- CLI: `spectramind.py --version` logs git SHA + config hash.
- `selftest` verifies files, config resolver, CLI registration.
- CI: Poetry env build, unit tests, smoke E2E.

**Phase 1 — Calibration & Data DAG**
- Kill chain implemented: ADC reversal → non-linearity → dark → flat → CDS/extract.
- Dry-run prints stage plan; full run writes DVC-tracked caches; negative values preserved.

**Phase 2 — Dual Encoders + Symbolic**
- `train --dry-run` validates shape flow and symbolic loss pack.
- Unit tests for smoothness & non-negativity constraints.
- Violation JSON written.

**Phase 3 — Curriculum & Calibration**
- Phases: MAE pretrain → optional contrastive → supervised GLL+λ·symbolic.
- `calibrate-temp` improves val GLL; `calibrate-corel` meets coverage.

**Phase 4 — Diagnostics & Dashboard**
- `diagnose dashboard` generates HTML with SHAP, UMAP/t-SNE, FFT, smoothness, symbolic overlays.
- CI publishes artifact.

**Phase 5 — Kaggle Runtime & Submit**
- Inference ≤30s/planet on staging; caching verified.
- `submit bundle` passes schema check, includes provenance & diagnostics HTML.

---

## 5) Performance Budgets

| Stage                | Target Time/Planet | Notes                                 |
|----------------------|--------------------|---------------------------------------|
| Calibration          | ≤10s               | DVC-cached, vectorized ops            |
| Encoding (FGS1+AIRS) | ≤5s                 | Batched GPU inference                 |
| Decoding (μ/σ)       | ≤2s                 | Single pass                           |
| Diagnostics          | ≤5s                 | Skip in Kaggle fast mode              |
| Total                | ≤30s                | Meets Kaggle ≤9h budget               |

---

## 6) Repo Layout (Authoritative) [oai_citation:6‡Repo and project structure start (North Star).pdf](file-service://file-48QwRwyruT8r78LkzfvJC3)

.
├── spectramind.py
├── configs/
│   ├── config_v50.yaml
│   ├── data/{local,kaggle}.yaml
│   ├── model/.yaml
│   ├── train/.yaml
│   └── diag/*.yaml
├── src/spectramind/
│   ├── data/
│   ├── models/
│   ├── symbolic/
│   ├── calibration/
│   ├── inference/
│   ├── training/
│   ├── diagnostics/
│   ├── cli/
│   └── utils/
├── outputs/
├── dvc.yaml
├── pyproject.toml / poetry.lock
├── Dockerfile
├── .github/workflows/ci.yml
├── v50_debug_log.md
└── events.jsonl

---

## 7) CLI & Hydra Usage

```bash
# Train with overrides
python -m spectramind train phase=supervised optimizer.lr=3e-4 model.fgs1_mamba.latent_dim=256

# Predict
python -m spectramind predict --out-csv outputs/submission.csv

# Calibrate
python -m spectramind calibrate-temp
python -m spectramind calibrate-corel

# Diagnose
python -m spectramind diagnose
python -m spectramind dashboard --html outputs/diagnostics/diag_v1.html

# Submit
python -m spectramind submit bundle


⸻

8) Operational Doctrine
	•	Branching: main is always release-ready; dev branches per feature; PRs require green CI + ADR if architecture changes.
	•	Reproducibility: no untracked artifacts; all datasets/checkpoints in DVC; all configs committed.
	•	Telemetry: every CLI call appends to v50_debug_log.md + events.jsonl with git SHA, config hash, env.
	•	Testing: unit tests for symbolic terms, calibration, CLI; integration tests for Phase 0–5 pipelines.

⸻

9) Risk Register & Mitigations

Risk	Mitigation
Calibration drift	Strict DVC stages, checksums, regression tests
σ miscalibration	Mandatory temp scaling + COREL coverage plots
FGS1 underfit	Increase Mamba capacity, add limb-darkening priors
Runtime overflow	Caching, batch inference, Kaggle fast mode


⸻

10) Roadmap
	•	Retrieval model integration (Bayesian inversion)
	•	LLM-assisted diagnostics (spectramind explain)
	•	Real molecular line databases (HITRAN, ExoMol) in symbolic loss
	•	3D latent visualization in HTML dashboard
	•	Deployment on real Ariel pipeline data post-2029

⸻

11) References
	1.	[North Star Execution Plan][354]
	2.	[Scientific Context & Symbolic Constraints][350]
	3.	[SpectraMindV50 Technical Design][6]
	4.	[NeurIPS Ariel Challenge AI Guide][7]
	5.	[Scientific References for NeurIPS 2025 Ariel Data Challenge][5]

---