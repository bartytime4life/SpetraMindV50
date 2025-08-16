SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

A neuro-symbolic, physics-informed pipeline for exoplanet transmission spectroscopy.
This repository is engineered for scientific rigor, full reproducibility, and fast iteration: Hydra configs, Typer CLI, DVC/lakeFS data discipline, MLflow-optional tracking, and a rich diagnostics dashboard.

⸻

✨ Highlights
•Dual-encoder architecture: FGS1 (Mamba SSM, long-sequence) + AIRS (edge-aware GNN) with multi-scale decoders for μ (mean) and σ (uncertainty).
•Neuro-symbolic constraints: Smoothness, non-negativity, FFT/asymmetry, photonic alignment; rule engines with violation maps and influence tracing.
•Uncertainty calibration: Temperature scaling + COREL (bin-wise conformal), with coverage/quantile checks and symbolic-region analysis.
•CLI-first workflow: One unified app (src/spectramind/spectramind.py) that runs training, inference, calibration, diagnostics, bundling, and log analysis.
•Reproducibility: Hydra YAMLs, config hashing, rotating logs + JSONL events, environment snapshots, optional Docker & CI.
•Diagnostics UI: Generate an interactive HTML dashboard (UMAP/t-SNE overlays, SHAP×symbolic fusion, GLL heatmaps, FFT, rule leaderboards).

⸻

📁 Repository Structure (core)

configs/                 # Hydra-safe YAMLs (model, calibration, diagnostics, hydra/*)
scripts/                 # One-liner wrappers: train/predict/calibrate/dashboard/bundle/etc.
src/spectramind/         # Library code (CLI, models, calibration, diagnostics, symbolic, etc.)
selftest.py              # Fast/deep repo consistency & integrity checks
README.md                # You are here
.gitignore               # Reproducibility-first ignore rules

Tip: all heavy artifacts (logs, checkpoints, datasets) are ignored by default for clean commits. Use DVC/lakeFS for data governance.

⸻

🚀 Quickstarts

0) Environment
•Python 3.10+ recommended.
•Optional: uv or poetry makes dependency management seamless.
•GPU/CUDA advised for training.

1) Self-Test

Run a fast integrity check (CLI registration, configs, shapes, and paths):

bash scripts/run_selftest.sh

Deep mode:

bash scripts/run_selftest.sh --deep --open-html

2) Train

bash scripts/train_v50.sh --config configs/model/config_v50.yaml --tag baseline

3) Predict & Bundle

bash scripts/predict_v50.sh --ckpt ckpts/best.pt --bundle --open-html

Or one-click end-to-end (selftest → predict → bundle):

bash scripts/one_click_make_submission.sh --ckpt ckpts/best.pt --open-html

4) Diagnostics Dashboard

bash scripts/diagnose_dashboard.sh --open

UMAP / t-SNE latents:

bash scripts/umap_latents.sh --open
bash scripts/tsne_latents.sh --open

Symbolic rule ranking:

bash scripts/symbolic_rank.sh

Symbolic profile overview:

bash scripts/profile_diagnose.sh --open


⸻

🧩 CLI Reference (top-level)

The unified Typer root is exposed as a Python module:

python -m src.spectramind.spectramind --help

Key subcommands (wrappers exist in scripts/):
•train — model training (Hydra configs, AMP, logging, hashing)
•predict — inference producing μ/σ + packaging hooks
•calibrate — photometry + σ calibration (temp scaling + COREL)
•diagnose — dashboard, UMAP/t-SNE, FFT, symbolic overlays, GLL/entropy
•analyze-log — parse v50_debug_log.md, export CSV/MD tables, heatmaps
•corel-train — dedicated COREL GNN training
•test — repository self-tests (selftest.py)

All commands write timestamped outputs under logs/… and append a config/run hash for reproducibility.

⸻

🔬 Reproducibility & Data Discipline
•Hydra: All runtime parameters live in YAML under configs/. Override with +key=value.
•Hashing: Every script writes run_hash.json and appends to run_hash_summary_v50.json.
•Data: Use DVC/lakeFS to version raw/calibrated datasets; keep Git clean.
•Env snapshots: bash scripts/export_env.sh --poetry --pip --conda --out .envsnap
•Docker: bash scripts/docker_build.sh --tag spectramind:v50

⸻

🧪 CI / Local Checks

Run a local CI mimic:

bash scripts/ci_check.sh --fast

This will run tests (if present), a fast selftest, and best-effort lint/format.

⸻

🧠 Scientific Notes
•Losses: Gaussian NLL (GLL) with spectral smoothness and asymmetry regularizers; optional quantile or diffusion decoders.
•Symbolic: Rule weights, violation masks, influence maps, and summaries exported to diagnostics.
•Calibration: Coverage and quantile evaluation per bin, molecule region summaries, and overlays in the dashboard.

⸻

🤝 Contributing
•Write code with docstrings, robust error handling, and clear logs.
•Keep configs Hydra-safe and minimal; add defaults to groups.
•Prefer functional commits and include selftest passes before pushing.
•Avoid committing large artifacts; use DVC/lakeFS remotes.

⸻

📜 License

Project-specific license TBD. If unsure, default to a permissive license (Apache-2.0 or MIT) suitable for research and challenge participation.

⸻

📧 Support

Open issues in the GitHub repository with:
•Command used, full console log excerpt
•Config overrides
•Environment snapshot (.envsnap/system_info.txt if available)

Stay stellar. ✨
