"""Experiment tracking behind a project-owned seam.

This is the only module that imports mlflow. Training code talks to a
tracker object, so swapping vendors or disabling tracking never touches
the training loop.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class NullTracker:
    """No-op tracker used when tracking is disabled (and in most tests)."""

    def log_params(self, params: dict) -> None:
        del params

    def set_tags(self, tags: dict) -> None:
        del tags

    def log_metrics(self, metrics: dict, step: int) -> None:
        del metrics, step

    def log_artifact(self, path: str | Path) -> None:
        del path

    def end(self, status: str = "FINISHED") -> None:
        del status


class MlflowTracker:
    """MLflow-backed tracker. Every mlflow call is fail-soft: a tracking
    failure prints one warning and never crashes training."""

    def __init__(
        self,
        run_name: str,
        *,
        experiment: str = "autodrive-rl",
        tracking_uri: str | None = None,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        self._warned = False
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)

    def _guard(self, operation) -> None:
        try:
            operation()
        except Exception as error:
            if not self._warned:
                self._warned = True
                print(f"warning: experiment tracking call failed: {error}")

    def log_params(self, params: dict) -> None:
        self._guard(
            lambda: self._mlflow.log_params(
                {key: str(value) for key, value in params.items()}
            )
        )

    def set_tags(self, tags: dict) -> None:
        if tags:
            self._guard(lambda: self._mlflow.set_tags(tags))

    def log_metrics(self, metrics: dict, step: int) -> None:
        numeric = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric:
            self._guard(lambda: self._mlflow.log_metrics(numeric, step=step))

    def log_artifact(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            self._guard(lambda: self._mlflow.log_artifact(str(path)))

    def end(self, status: str = "FINISHED") -> None:
        self._guard(lambda: self._mlflow.end_run(status=status))


def git_commit_tag() -> dict:
    """Best-effort {"git_commit": <short-hash>}; empty dict if unavailable."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return {"git_commit": commit} if commit else {}
    except Exception:
        return {}


def create_tracker(
    enabled: bool,
    run_name: str,
    *,
    tracking_uri: str | None = None,
) -> NullTracker | MlflowTracker:
    """Build the tracker for a training run."""

    if not enabled:
        return NullTracker()
    try:
        import mlflow  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "mlflow is not installed. Run: python -m pip install -r "
            "requirements.txt, or pass --no-tracking."
        ) from error
    return MlflowTracker(run_name, tracking_uri=tracking_uri)
