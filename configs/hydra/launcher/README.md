# Hydra Launcher Group — SpectraMind V50

This directory defines the Hydra **launcher group** for SpectraMind V50. Launchers control how and where jobs run (local sequential, local parallel, SLURM, Kaggle). Selecting a launcher changes only the execution environment — not experiment semantics — and all runs remain reproducible with consistent logging and config snapshots.

## Files

* `default.yaml` — entry point that chooses the active launcher (override at CLI).
* `basic.yaml` — local sequential launcher (safe for debugging/CI).
* `local.yaml` — local parallel multirun using Hydra’s BasicLauncher with `max_parallel_jobs`.
* `slurm.yaml` — HPC cluster launcher targeting SLURM via Submitit.
* `kaggle.yaml` — Kaggle GPU runtime profile with 9-hour guardrails and env tagging.

## Run output layout (same for all launchers)

* Single runs  → `logs/runs/YYYY-MM-DD/HH-MM-SS/`
* Multiruns    → `logs/multiruns/YYYY-MM-DD/HH-MM-SS/`
* Hydra state  → `.hydra/` (merged config snapshot, `overrides.yaml`, `hydra.yaml`)

## Quick start

* Sequential local:
  `python -m spectramind.cli.spectramind train hydra/launcher=basic`
* Parallel local (e.g., 4 jobs):
  `python -m spectramind.cli.spectramind ablate -m hydra/launcher=local hydra.launcher.max_parallel_jobs=4`
* SLURM (one GPU, 32GB, 8 CPUs; override as needed):
  ```
  python -m spectramind.cli.spectramind train -m  
  hydra/launcher=slurm  
  hydra.launcher.partition=gpu  
  hydra.launcher.gres=gpu:1  
  hydra.launcher.cpus_per_task=8  
  hydra.launcher.mem_gb=32  
  hydra.launcher.timeout_min=600
  ```
* Kaggle GPU runtime (sequential by platform constraints):
  `python -m spectramind.cli.spectramind submit hydra/launcher=kaggle`

## Reproducibility & logging
Each run records:

* `.hydra/config.yaml` (full merged config), `overrides.yaml`, `hydra.yaml`
* Console + rotating file logs; optional JSONL event stream
* ENV & Git metadata via SpectraMind runtime hooks
* Optional MLflow/W&B sync if enabled

Reproduce a past run by reusing `.hydra/config.yaml` or `.hydra/overrides.yaml` with the same environment (Poetry/Docker) and data/DVC state.

## Local parallelism
`local.yaml` exposes `max_parallel_jobs`:
`hydra/launcher=local hydra.launcher.max_parallel_jobs=8`
Mind GPU contention; set per-run device affinity (e.g., `CUDA_VISIBLE_DEVICES`) if needed.

## SLURM (Submitit) notes
Key fields in `slurm.yaml`:

* `partition`, `gres`, `cpus_per_task`, `mem_gb`, `timeout_min`, `max_num_timeout`
* `submitit_folder` for job metadata; `name` for SLURM job names
  Useful overrides:
  `hydra.launcher.nodelist=nodeA`
  `hydra.launcher.qos=high`
  `hydra.launcher.account=my_lab`
  `hydra.launcher.array_parallelism=4`
  Ensure `hydra-submitit-launcher` is installed.

## Kaggle notes

* Single GPU, sequential execution, strict wall-time
* Avoid large `-m` sweeps; run curated configs
* Keep artifacts under working directory; be mindful of disk limits
* Use lightweight logs/HTML exports

## Change the default launcher
Edit `configs/hydra/launcher/default.yaml`:
```
defaults:
- override hydra/launcher: basic
```
Or override at CLI: `hydra/launcher=slurm`

## Troubleshooting

* SLURM jobs idle: check partition/account/qos; reduce `mem_gb`/`cpus_per_task`
* CUDA OOM: lower batch size; serialize jobs (`max_parallel_jobs=1`)
* Log write issues: confirm `logs/` is writable; paths are relative to CWD
* Too many sweep runs: shrink grids or use random sampling; control SLURM array parallelism

## Minimal examples

* Single run (default/basic):
  `python -m spectramind.cli.spectramind train`
* Explicit basic:
  `python -m spectramind.cli.spectramind train hydra/launcher=basic`
* Local parallel dashboard (4 workers):
  ```
  python -m spectramind.cli.spectramind diagnose dashboard -m  
  hydra/launcher=local hydra.launcher.max_parallel_jobs=4  
  diagnostics.umap=true diagnostics.tsne=true
  ```
* SLURM long job:
  ```
  python -m spectramind.cli.spectramind ablate -m  
  hydra/launcher=slurm  
  hydra.launcher.partition=gpu  
  hydra.launcher.gres=gpu:1  
  hydra.launcher.mem_gb=48  
  hydra.launcher.timeout_min=1200  
  ablate.top_n=20
  ```
* Kaggle submission:
  `python -m spectramind.cli.spectramind submit hydra/launcher=kaggle`

## Best practices

* Keep resource requests in config; override via CLI only when probing capacity
* Always inspect `.hydra/` snapshots to reproduce exact runs
* Prefer `basic.yaml` for CI to minimize variability
* Use SLURM arrays for large sweeps and throttle with `array_parallelism`
