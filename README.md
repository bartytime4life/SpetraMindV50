SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge (Mission‑Grade Root)

SpectraMind V50 is a neuro‑symbolic, physics‑informed AI pipeline engineered to excel in the NeurIPS 2025 Ariel Data Challenge. It is designed to be a reproducible, CLI-first, ESA‑Ariel‑aware, NASA‑grade research system with strict logging, experiment tracking, and symbolic rule integration that encodes physical truths directly into learning.

Mission Brief
•Objective: Predict mean (μ) and uncertainty (σ) spectra for exoplanet atmospheres from dual instruments (FGS1 temporal, AIRS spectral), with leaderboard‑grade Generalized Log Likelihood (GLL) performance under strict runtime constraints (≤9 hours across ~1,100 planets).
•Strategy:
•FGS1-first temporal modeling (Mamba SSM or equivalent long‑sequence encoder) to extract robust transit dynamics.
•AIRS spectral GNN (e.g., GAT/RGCN/edge‑aware) with wavelength‑graph structure, molecule/region edges, and symbolic priors.
•Multi‑scale μ decoder + Flow σ head (calibration-aware), with symbolic constraints (smoothness, non‑negativity, molecular coherence).
•Calibration kill chain for photometric corrections and alignment; COREL conformal coverage tuning; temperature scaling.
•Diagnostics and Explainability: SHAP overlays, symbolic rule violation dashboards, FFT/UMAP/t‑SNE overlays, and HTML reports.

Reproducibility & Engineering Tenets
•CLI-first: A single Typer multiplexer (spectramind.py) orchestrates training, inference, calibration, diagnostics, dashboard, ablations, and submission.
•Hydra configs: All parameters are composable YAML; never hardcode in code paths.
•Logging discipline:
•Console logs (human‑readable).
•Rotating file logs (logs/…), retention guarded.
•JSONL event stream (logs/events.jsonl) for machine auditability.
•Experiment tracking: MLflow by default; optional W&B.
•Git/ENV capture: Each run records Git SHA/dirty flag, diff summary, pip freeze, CUDA/cudnn info, and host specs to run_hash_summary_v50.json.
•CI/Policy Gates: GitHub Actions run selftests, diagnostics, docs build, and submission sanity checks. Optional OPA/Conftest for K8s/Helm if deployed.
•Data governance: Data paths via Hydra only; no path literals in code; DVC/lakeFS recommended (configured in sub‑scaffold).

This README is the root-level handbook. See docs/ and ARCHITECTURE.md for deep diagrams, module contracts, and acceptance tests.

⸻

Quickstart (once full scaffold is in place)

poetry install --no-root
poetry run python -m spectramind --version
poetry run python -m spectramind selftest --fast
poetry run python -m spectramind diagnose dashboard --no-open
poetry run python -m spectramind submit make --dry-run

Logs & Reports
•Human log: logs/spectramind.log (rotating)
•JSONL events: logs/events.jsonl
•Run hash: run_hash_summary_v50.json
•Diagnostics HTML: artifacts/diagnostics/index.html

⸻

Root Files in This Repository

This root contains (created by scaffold_root_spectramindv50.sh):
•README.md (this file)
•ARCHITECTURE.md (engineering spec; if missing, generate from docs scaffold)
•LICENSE (MIT by default)
•CHANGELOG.md (Keep a changelog)
•CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
•CITATION.cff
•VERSION (semantic version for CLI/reporting)
•.gitignore, .dockerignore, .gitattributes, .editorconfig
•.pre-commit-config.yaml
•pyproject.toml (Poetry/ruff/pytest/black/isort/mkdocs config root)
•mkdocs.yml (Docs site definition)
•Dockerfile (GPU-ready base stub; extend to your infra)
•.vscode/settings.json (editor hygiene)
•push_root_files.sh (one-liner to commit/push these root files)

Subdirectories like src/, configs/, scripts/, docs/, tests/ are created by dedicated scaffolders. This root scaffold only asserts the top-level contract.

⸻

Governance & Safety
•Pre-commit enforces style and security checks.
•Security policy describes vulnerability reporting.
•Code of Conduct sets community expectations.
•License defaults to MIT; adjust if needed.

⸻

Citation

If this work informs your research, please cite the repository via CITATION.cff.

⸻

Support
•Issues: use GitHub Issues with labeled templates.
•Discussions: roadmap and research threads welcome.
•PRs: must pass CI, selftests, and lint/policy gates.

— SpectraMind V50 (master‑architect edition)
