# Hydra Sweeper Configs — SpectraMind V50

This directory contains hyperparameter sweeper configurations for SpectraMind V50 using Hydra plugin groups.

## Files

* `optuna_tpe.yaml` — Optuna TPE sampler with deterministic seed, median pruner, and a challenge-grade search space spanning optimizer, schedule, capacities, and symbolic losses.
* `optuna_cmaes.yaml` — Optuna CMA-ES sampler for smooth continuous landscapes.
* `optuna_random.yaml` — Optuna Random sampler for broad exploration and smoke tests.
* `optuna_nsga2.yaml` — Multi-objective Optuna NSGA-II (minimize GLL and calibration error).
* `basic.yaml` — Hydra Basic sweeper for small deterministic grids.

## Typical Commands

Local TPE:

```bash
python -m spectramind.cli.spectramind train -m hydra/sweeper=optuna_tpe
```

Slurm + TPE:

```bash
python -m spectramind.cli.spectramind train -m hydra/launcher=submitit_slurm hydra/sweeper=optuna_tpe +hydra.sweeper.n_trials=200
```

Local CMA-ES:

```bash
python -m spectramind.cli.spectramind train -m hydra/sweeper=optuna_cmaes
```

Multi-objective (NSGA-II):

```bash
python -m spectramind.cli.spectramind train -m hydra/sweeper=optuna_nsga2
```

Basic (grid-like discrete expansion):

```bash
python -m spectramind.cli.spectramind train -m hydra/sweeper=basic training.batch_size=16,32 model.encoder_dim=384,512
```

## Notes

* Override any parameter from CLI; sweeper `params` act as defaults.
* For distributed sweeps, set `hydra.sweeper.storage` to a shared DB (e.g., `sqlite:///optuna.db` or a Postgres URI).
* Combine with any launcher in `configs/hydra/launcher`.
