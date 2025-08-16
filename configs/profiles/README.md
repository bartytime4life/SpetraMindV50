# configs/profiles — Scenario Bundles (Hydra)

These profiles are **Hydra override bundles** that select concrete group configs for:

* `model` (e.g., `configs/model/v50.yaml`)
* `calibration/steps` (e.g., `minimal`, `standard`, `full`, `corel_plus_temp`, `symbolic_weighted`)
* `diagnostics` (e.g., `basic`, `lite`, `leaderboard`, `symbolic`, `explain`, `calibration`)
* `logging` (e.g., `standard`, `debug`, `mission`, `symbolic`, `verbose`)
* `symbolic/profiles` (e.g., `default`, `strict`, `relaxed`, `exploratory`)

Available profiles:

* `default.yaml` — balanced defaults for research/testing, reproducible.
* `fast_dev.yaml` — faster runs; minimal calibration & diagnostics, debug logs.
* `kaggle_leaderboard.yaml` — mission-grade rigor for challenge submissions (≤9h).
* `symbolic_strict.yaml` — hard enforcement of symbolic rules.
* `explainability.yaml` — SHAP, UMAP, t-SNE, overlays & dashboards.
* `calibration_heavy.yaml` — COREL + temperature scaling focus, coverage-first.

### Usage

Activate any profile with:

* Python module:
  * `python -m spectramind.cli.spectramind train profiles=kaggle_leaderboard`
  * `python -m spectramind.cli.spectramind diagnose dashboard profiles=explainability`
* Direct script:
  * `python src/spectramind/train_v50.py profiles=symbolic_strict`
  * `python src/spectramind/predict_v50.py profiles=default`

The active profile name is exposed as `${active_profile}` (from Hydra runtime), which is embedded in log paths and context.

### Notes

* Profiles merge with `configs/config.yaml`; they can override `reproducibility`, `project.mode`, etc.
* Groups are designed to be orthogonal and composable.
* Add new profiles by composing `defaults` with existing or new group files.
