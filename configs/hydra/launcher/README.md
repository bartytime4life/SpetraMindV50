# Hydra Launchers — SpectraMind V50

Select a launcher per environment:

* `submitit_slurm.yaml` — Slurm clusters (multi-node/GPU).
  Example:

  ```bash
  python -m spectramind.cli.spectramind train -m hydra/launcher=submitit_slurm hydra/sweeper=optuna_tpe +hydra.sweeper.n_trials=200
  ```

* `submitit_local.yaml` — Submitit local backend for dev parity.

  ```bash
  python -m spectramind.cli.spectramind train -m hydra/launcher=submitit_local hydra/sweeper=optuna_random
  ```

* `joblib.yaml` — Local multi-process parallelism.

  ```bash
  python -m spectramind.cli.spectramind train -m hydra/launcher=joblib hydra/sweeper=basic training.batch_size=16,32
  ```

* `ray.yaml` — Ray cluster or local Ray session.

  ```bash
  python -m spectramind.cli.spectramind train -m hydra/launcher=ray hydra/sweeper=optuna_cmaes
  ```

* `basic.yaml` — Built-in Hydra launcher (in-process).

  ```bash
  python -m spectramind.cli.spectramind train -m hydra/launcher=basic hydra/sweeper=optuna_tpe
  ```

Best practices:

* Combine any `hydra/launcher=*` with any `hydra/sweeper=*`.
* Tune `array_parallelism` / `n_jobs` to avoid overloading nodes.
* For distributed Optuna, set `hydra.sweeper.storage` to a shared DB (e.g., `sqlite:///optuna.db`).
