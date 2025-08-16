# Hydra Launcher Group — SpectraMind V50

This directory defines the **Hydra launcher group** for SpectraMind V50. Launchers control **how** and **where** jobs run (local, parallel, SLURM, Kaggle). Selecting a launcher never changes your experiment semantics — only the execution environment — and all runs remain **reproducible** with consistent logging and config snapshots.

---

## Files

* `default.yaml` — entry point that chooses the active launcher (overridable at the CLI).
* `basic.yaml` — default **local sequential** launcher (safe for debugging and CI).
* `local.yaml` — **local parallel** multirun using Hydra’s basic launcher with `max_parallel_jobs`.
* `slurm.yaml` — **HPC cluster** launcher targeting SLURM via Submitit.
* `kaggle.yaml` — **Kaggle GPU** runtime profile with 9-hour guardrails and env tagging.

All launchers write:

* Single runs → `logs/runs/YYYY-MM-DD/HH-MM-SS/`
* Multiruns   → `logs/multiruns/YYYY-MM-DD/HH-MM-SS/`
* Hydra state → `.hydra/` (config, overrides, and `hydra.yaml` snapshots)

These locations are controlled by `hydra.run.dir`, `hydra.sweep.dir`, and `hydra.output_subdir` in `default.yaml`.

---

## Quick Start

### Pick a launcher at runtime

* Sequential local:

  ```
  python -m spectramind.cli.spectramind train hydra/launcher=basic
  ```
* Parallel local (e.g., 4 jobs):

  ```
  python -m spectramind.cli.spectramind ablate hydra/launcher=local hydra.launcher.max_parallel_jobs=4
  ```
* SLURM (one GPU, 32GB, 8 CPUs; edit as needed or override at CLI):

  ```
  python -m spectramind.cli.spectramind train \
    hydra/launcher=slurm \
    hydra.launcher.partition=gpu \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.mem_gb=32 \
    hydra.launcher.timeout_min=600
  ```
* Kaggle GPU runtime (sequential by platform constraints):

  ```
  python -m spectramind.cli.spectramind submit hydra/launcher=kaggle
  ```

> Tip: You can also change the default by editing `defaults` in `default.yaml`.

---

## Common Patterns

### 1) Multirun (sweeps) with parallel local launcher

```
python -m spectramind.cli.spectramind train \
  -m hydra/launcher=local hydra.launcher.max_parallel_jobs=6 \
  model.lr=1e-4,2e-4,5e-4 seed=7,11,13
```

Hydra creates one subfolder per combination under `logs/multiruns/.../` and saves a `.hydra/` state per job.

### 2) SLURM array-style sweeps

Submitit handles array-like multiruns transparently when you pass `-m`:

```
python -m spectramind.cli.spectramind train \
  -m hydra/launcher=slurm \
  hydra.launcher.partition=gpu \
  hydra.launcher.gres=gpu:1 \
  seed=0,1,2
```

Logs and `.hydra/` snapshots for each task are kept under the sweep directory; Submitit metadata is stored in `${hydra.sweep.dir}/.slurm_submitit`.

### 3) Kaggle runtime guardrails

Kaggle provides a single GPU with a hard wall-time. The `kaggle.yaml` profile:

* sets `env.KAGGLE=true` and `env.KAGGLE_GPU=1` for programmatic checks,
* recommends `timeout_min=540` (9 hours),
* runs sequentially (no parallel sweeps).

Example:

```
python -m spectramind.cli.spectramind submit \
  hydra/launcher=kaggle \
  pipeline.leaderboard_mode=true \
  runtime.max_duration_min=520
```

---

## Reproducibility & Logging

Each run records:

* **Effective configs** in `.hydra/`:

  * `config.yaml` — full merged config for the run
  * `overrides.yaml` — CLI overrides
  * `hydra.yaml` — Hydra’s own resolved config
* **Console logs** + **rotating file logs** (see project logging package)
* **JSONL event stream** (if enabled by logging config)
* **ENV & Git metadata** (captured by SpectraMind’s runtime hooks)
* Optional MLflow/W&B sync (respects your config flags)

Reproducing a run is as simple as:

1. `cd` into the run directory.
2. Reuse the saved `overrides.yaml` or the full `config.yaml` with your CLI.
3. Ensure the same environment (Python/Poetry/Docker) and data cache/DVC state.

---

## SLURM Notes (Submitit)

* **Key fields** in `slurm.yaml`:

  * `partition`, `gres`, `cpus_per_task`, `mem_gb`, `timeout_min`, `max_num_timeout`
  * `submitit_folder` to keep Submitit’s pickles and job descriptors
  * `name` to tag SLURM job names
* **Useful overrides**:

  ```
  hydra.launcher.nodelist=nodeA
  hydra.launcher.qos=high
  hydra.launcher.account=my_lab
  hydra.launcher.array_parallelism=4
  ```
* Make sure `hydra-submitit-launcher` is installed in your environment.

---

## Kaggle Notes

* Single GPU, sequential execution, tight wall time.
* Avoid large `-m` sweeps; instead, run curated configs.
* Ensure all artifacts write to the working directory and that the submission bundle is kept under Kaggle’s output path.
* Prefer lightweight logs and HTML exports to fit disk limits.

---

## Local Parallelism

`local.yaml` exposes `max_parallel_jobs`. Example:

```
hydra/launcher=local hydra.launcher.max_parallel_jobs=8
```

Use with care if your experiments require exclusive GPU access. Combine with per-run device affinity (e.g., `CUDA_VISIBLE_DEVICES`) if needed.

---

## Changing the Default Launcher

Edit `configs/hydra/launcher/default.yaml`:

```
defaults:
  - override hydra/launcher: basic   # change to local | slurm | kaggle
```

Or override at the CLI: `hydra/launcher=slurm`.

---

## Troubleshooting

* **Jobs queued but not running (SLURM)**: Check `partition`, `account`, `qos`, and cluster policies. Reduce `mem_gb` or `cpus_per_task` if constrained.
* **CUDA OOM**: Lower batch size or ensure jobs are serialized per GPU. On local parallel runs, set `max_parallel_jobs=1` for exclusive access.
* **Permission denied writing logs**: Confirm `logs/` is writable. The default paths are relative to the CWD that launches the run.
* **Hydra sweep too many runs**: Use smaller grids, random sampling (if enabled in your sweep logic), or increase `array_parallelism` on SLURM.

---

## Minimal Examples

* Single run (default/basic):

  ```
  python -m spectramind.cli.spectramind train
  ```
* Explicit basic:

  ```
  python -m spectramind.cli.spectramind train hydra/launcher=basic
  ```
* Local parallel (4 workers):

  ```
  python -m spectramind.cli.spectramind diagnose dashboard -m \
    hydra/launcher=local hydra.launcher.max_parallel_jobs=4 \
    diagnostics.umap=true diagnostics.tsne=true
  ```
* SLURM long job:

  ```
  python -m spectramind.cli.spectramind ablate -m \
    hydra/launcher=slurm \
    hydra.launcher.partition=gpu \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.mem_gb=48 \
    hydra.launcher.timeout_min=1200 \
    ablate.top_n=20
  ```
* Kaggle submission:

  ```
  python -m spectramind.cli.spectramind submit hydra/launcher=kaggle
  ```

---

## Best Practices

* Keep launcher-specific resource requests **in the config** and only override **at the CLI** when probing capacity.
* Always check the saved `.hydra/` folder to reproduce exact runs.
* For CI, prefer `basic.yaml` to reduce variability.
* For large sweeps, use SLURM arrays via `-m` and control parallelism with cluster policy and launcher parameters.

---

## Extending Launchers

You can add more launchers (e.g., PBS, LSF, Ray) by adding a new YAML (e.g., `ray.yaml`) and pointing `_target_` to the corresponding Hydra plugin. Then select it via `hydra/launcher=ray` at the CLI.
