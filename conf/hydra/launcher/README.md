# Hydra Launcher configs

This directory configures how SpectraMind jobs are launched via [Hydra](https://hydra.cc/).
Each YAML file below can be selected with Hydra's group override syntax
(e.g. `python spectramind.py hydra/launcher=slurm`).

## Available launchers

| File | Description | Usage |
| ---- | ----------- | ----- |
| `default.yaml` | Sets run/sweep logging directories and selects `basic` launcher by default. | n/a (imported automatically) |
| `basic.yaml` | Sequential local execution; safe for debugging and tests. | `hydra/launcher=basic` |
| `local.yaml` | Local launcher with configurable parallelism via `max_parallel_jobs`. | `hydra/launcher=local` |
| `slurm.yaml` | SLURM cluster launcher using `hydra-submitit` with GPU, CPU and memory resources. | `hydra/launcher=slurm` |
| `kaggle.yaml` | Kaggle competition runtime (single GPU, 9h limit, Kaggle env vars). | `hydra/launcher=kaggle` |

Run without specifying a launcher to use the `basic` sequential mode by default.
