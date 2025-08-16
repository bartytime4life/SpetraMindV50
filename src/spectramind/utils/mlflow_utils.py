from typing import Any, Dict, Optional


class MlflowClientSafe:
    """Optional MLflow client interface that is a no-op if MLflow is not installed."""

    def __init__(self) -> None:
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
        except Exception:
            self._mlflow = None

    def start_run(self, run_name: Optional[str] = None):
        if self._mlflow is None:
            return None
        return self._mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        if self._mlflow is None:
            return
        self._mlflow.end_run()

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if self._mlflow is None:
            return
        self._mlflow.set_tags(tags)


def mlflow_safe_log_params(params: Dict[str, Any]) -> None:
    try:
        import mlflow  # type: ignore

        mlflow.log_params(params)
    except Exception:
        pass


def mlflow_safe_log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    try:
        import mlflow  # type: ignore

        mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass
