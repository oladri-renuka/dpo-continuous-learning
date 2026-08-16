"""MLflow client for experiment tracking and model registry."""

from typing import Dict, Any, Optional, List
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
import mlflow
from mlflow.entities import Run

log = structlog.get_logger(__name__)


class MLflowClientError(Exception):
    """Custom exception for MLflow operations."""
    pass


class MLflowClient:
    """
    MLflow client for logging metrics, params, and artifacts.
    Handles connection failures gracefully with retries.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "default",
        run_name: Optional[str] = None,
    ):
        """
        Initialize MLflow client (production only).

        Args:
            tracking_uri: MLflow tracking server URI (default: from environment or localhost:5000)
            experiment_name: Name of the experiment to log to
            run_name: Optional name for this run

        Raises:
            MLflowClientError: If MLflow server is unreachable
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.experiment_name = experiment_name
        self.run_name = run_name

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            log.info("MLflow tracking URI set", uri=self.tracking_uri)
        except Exception as e:
            log.error("Failed to set MLflow tracking URI", error=str(e))
            raise MLflowClientError(f"MLflow server unreachable at {self.tracking_uri}: {str(e)}") from e

        self.current_run: Optional[Run] = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def start_run(self, experiment_name: Optional[str] = None, run_name: Optional[str] = None) -> str:
        """
        Start a new MLflow run.

        Args:
            experiment_name: Override default experiment name
            run_name: Optional human-readable run name

        Returns:
            Run ID

        Raises:
            MLflowClientError: If run creation fails
        """
        try:
            exp_name = experiment_name or self.experiment_name
            r_name = run_name or self.run_name

            # Set or create experiment
            mlflow.set_experiment(exp_name)
            log.info("MLflow experiment set", experiment=exp_name)

            # Start run
            self.current_run = mlflow.start_run(run_name=r_name)
            run_id = self.current_run.info.run_id
            log.info("MLflow run started", run_id=run_id, run_name=r_name)
            return run_id
        except Exception as e:
            log.error("Failed to start MLflow run", error=str(e))
            raise MLflowClientError(f"Failed to start MLflow run: {str(e)}") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params: Dictionary of parameter names and values

        Raises:
            MLflowClientError: If logging fails
        """
        # Convert non-string values to strings
        params_str = {str(k): str(v) for k, v in params.items()}
        mlflow.log_params(params_str)
        log.debug("Logged parameters to MLflow", param_count=len(params))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step counter

        Raises:
            MLflowClientError: If logging fails
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value), step=step)
        log.debug("Logged metrics to MLflow", metric_count=len(metrics))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact (file or directory).

        Args:
            local_path: Local path to file or directory
            artifact_path: Optional path within MLflow artifact storage

        Raises:
            MLflowClientError: If logging fails
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        log.info("Logged artifact to MLflow", local_path=local_path, artifact_path=artifact_path)

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.

        Args:
            status: Status code ('FINISHED' or 'FAILED')

        Raises:
            MLflowClientError: If ending run fails
        """
        mlflow.end_run(status=status)
        log.info("MLflow run ended", status=status)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def get_latest_run(self, experiment_name: str) -> Optional[Run]:
        """
        Get the latest run from an experiment.

        Args:
            experiment_name: Name of experiment

        Returns:
            Latest run, or None if no runs found

        Raises:
            MLflowClientError: If query fails
        """
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            log.warning("Experiment not found", experiment=experiment_name)
            return None

        # Get latest run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            run = mlflow.get_run(run_id)
            log.info("Retrieved latest run", experiment=experiment_name, run_id=run_id)
            return run

        log.warning("No runs found in experiment", experiment=experiment_name)
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def log_model(
        self,
        model_path: str,
        artifact_path: str = "model",
    ) -> None:
        """
        Log a model to MLflow Model Registry.

        Args:
            model_path: Local path to model directory
            artifact_path: Path within artifact storage

        Raises:
            MLflowClientError: If logging fails
        """
        mlflow.log_artifact(model_path, artifact_path=artifact_path)
        log.info("Logged model to MLflow", model_path=model_path, artifact_path=artifact_path)

    def get_current_run_id(self) -> Optional[str]:
        """Get the current active run ID."""
        if self.current_run:
            return self.current_run.info.run_id
        return mlflow.active_run().info.run_id if mlflow.active_run() else None
