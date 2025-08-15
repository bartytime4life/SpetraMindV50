from typing import Any, Dict, Optional

# MLflow is optional; delay import to runtime and no-op if unavailable.


class MLflowSync:
    """
    Minimal MLflow wrapper that degrades gracefully if mlflow is not installed.
    """

    def __init__(self, experiment_name: str = "SpectraMindV50"):
        self._ok = False
        self._mlflow = None
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
            self._mlflow.set_experiment(experiment_name)
            self._ok = True
        except Exception:
            self._ok = False
            self._mlflow = None
        self.active_run = None

    def start_run(self, run_name: Optional[str] = None):
        if not self._ok:
            return None
        self.active_run = self._mlflow.start_run(run_name=run_name)
        return self.active_run

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._ok:
            self._mlflow.log_params(params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if self._ok:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        if self._ok:
            self._mlflow.log_artifact(path)

    def end_run(self) -> None:
        if self._ok:
            self._mlflow.end_run()
