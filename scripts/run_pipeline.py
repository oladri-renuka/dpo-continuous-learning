"""Orchestrator: chains all pipeline steps (Aggregator → Trainer → Quality Gate → Deployer)."""

import sys
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.infra import setup_logging, get_logger
from src.infra.mlflow_client import MLflowClient
from src.core.aggregator import Aggregator, AggregationError
from src.core.trainer import DPOTrainer, TrainerError
from src.core.quality_gate import QualityGate, ModelDegradationError
from src.api.deployment import deploy_model

log = get_logger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the entire nightly DPO learning loop.

    Pipeline Flow:
    1. Aggregator: Consume Kafka → create train/val datasets
    2. Baseline Check: Validate data quality (data must be learnable)
    3. Trainer: QLoRA + DPO training on RunPod
    4. Quality Gate: Validate metrics (reward acc > 72%, win-rate > 55%)
    5. Deployer: Blue/Green deployment with canary rollout
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.mlflow_client = MLflowClient()
        self.start_time = time.time()

    def run(self) -> int:
        """
        Execute the complete pipeline.

        Returns:
            Exit code (0 = success, 1 = failure)
        """
        log.info("=" * 80)
        log.info("ORCHESTRATOR: Starting DPO nightly pipeline")
        log.info("=" * 80)

        # Start MLflow run for the entire pipeline
        try:
            run_id = self.mlflow_client.start_run(
                experiment_name="nightly-pipeline",
                run_name=f"pipeline-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            )
            log.info("MLflow run started", run_id=run_id)
        except Exception as e:
            log.warning("Failed to start MLflow run (non-blocking)", error=str(e))

        try:
            # ===================================================================
            # STEP 1: AGGREGATOR (Consume Kafka → Train/Val datasets)
            # ===================================================================
            log.info("\n" + "=" * 80)
            log.info("STEP 1: AGGREGATOR")
            log.info("=" * 80)

            aggregator = Aggregator()
            train_path, val_path = aggregator.aggregate_feedback()

            if not train_path or not val_path:
                raise AggregationError("Aggregator did not produce train/val paths")

            log.info("✓ Aggregation successful", train=train_path, val=val_path)

            # ===================================================================
            # STEP 2: BASELINE CHECK (Data quality gate)
            # ===================================================================
            log.info("\n" + "=" * 80)
            log.info("STEP 2: BASELINE CHECK")
            log.info("=" * 80)

            from scripts.baseline_check import run_baseline_check

            baseline_acc = run_baseline_check(train_data_path=train_path)
            if baseline_acc < 0.55:
                log.error(
                    "DATA INTRINSICALLY NOISY",
                    baseline_accuracy=f"{baseline_acc:.2%}",
                    threshold="0.55",
                )
                try:
                    self.mlflow_client.log_params(
                        {
                            "pipeline_step": "baseline_check",
                            "status": "FAILED",
                            "baseline_accuracy": baseline_acc,
                        }
                    )
                    self.mlflow_client.end_run(status="FAILED")
                except Exception as mlflow_err:
                    log.warning("Failed to log to MLflow (non-blocking)", error=str(mlflow_err))
                return 1

            log.info("✓ Baseline check passed", accuracy=f"{baseline_acc:.2%}")

            # ===================================================================
            # STEP 3: TRAINER (QLoRA + DPO on RunPod)
            # ===================================================================
            log.info("\n" + "=" * 80)
            log.info("STEP 3: TRAINER (QLoRA + DPO)")
            log.info("=" * 80)

            trainer = DPOTrainer(train_data_path=train_path, val_data_path=val_path)
            adapter_path = trainer.train()

            log.info("✓ Training complete", adapter_path=adapter_path)

            # ===================================================================
            # STEP 4: QUALITY GATE (Metric validation - HARD STOP)
            # ===================================================================
            log.info("\n" + "=" * 80)
            log.info("STEP 4: QUALITY GATE (METRIC VALIDATION)")
            log.info("=" * 80)

            gate = QualityGate(
                challenger_adapter_path=adapter_path,
                champion_adapter_path="s3://ml-artifacts/models/champion/adapter_model.bin",
            )

            try:
                gate.validate()
                log.info(
                    "✓ Quality gate passed",
                    reward_accuracy=f"{gate.metrics.reward_accuracy:.3f}",
                    dpo_win_rate=f"{gate.metrics.dpo_win_rate:.3f}",
                )
            except ModelDegradationError as e:
                log.error("QUALITY GATE FAILED", error=str(e))
                try:
                    self.mlflow_client.log_params(
                        {
                            "pipeline_step": "quality_gate",
                            "status": "FAILED",
                            "error": str(e),
                        }
                    )
                    self.mlflow_client.end_run(status="FAILED")
                except Exception as mlflow_err:
                    log.warning("Failed to log to MLflow (non-blocking)", error=str(mlflow_err))
                return 1

            # ===================================================================
            # STEP 5: DEPLOYER (Blue/Green with canary rollout)
            # ===================================================================
            log.info("\n" + "=" * 80)
            log.info("STEP 5: DEPLOYER (BLUE/GREEN DEPLOYMENT)")
            log.info("=" * 80)

            deployment_result = deploy_model(
                challenger_adapter_path=adapter_path,
                challenger_version=f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            )

            if deployment_result["status"] != "success":
                log.error("Deployment failed", reason=deployment_result.get("reason"))
                try:
                    self.mlflow_client.log_params(
                        {
                            "pipeline_step": "deployer",
                            "status": "FAILED",
                            "reason": deployment_result.get("reason"),
                        }
                    )
                    self.mlflow_client.end_run(status="FAILED")
                except Exception as mlflow_err:
                    log.warning("Failed to log to MLflow (non-blocking)", error=str(mlflow_err))
                return 1

            log.info("✓ Deployment successful", version=deployment_result["champion_version"])

            # ===================================================================
            # SUCCESS: Complete pipeline
            # ===================================================================
            duration = (time.time() - self.start_time) / 3600

            log.info("\n" + "=" * 80)
            log.info("✓✓✓ NIGHTLY PIPELINE COMPLETE ✓✓✓")
            log.info("=" * 80)
            log.info(
                "Pipeline summary",
                total_duration_hours=f"{duration:.2f}",
                new_champion=deployment_result["champion_version"],
                baseline_accuracy=f"{baseline_acc:.2%}",
                reward_accuracy=f"{gate.metrics.reward_accuracy:.3f}",
                dpo_win_rate=f"{gate.metrics.dpo_win_rate:.3f}",
            )

            try:
                self.mlflow_client.log_params(
                    {
                        "pipeline_status": "SUCCESS",
                        "total_duration_hours": f"{duration:.2f}",
                    }
                )
                self.mlflow_client.end_run(status="FINISHED")
            except Exception as mlflow_err:
                log.warning("Failed to log to MLflow (non-blocking)", error=str(mlflow_err))

            return 0

        except AggregationError as e:
            log.error("Aggregation failed (HARD STOP)", error=str(e))
            try:
                self.mlflow_client.log_params(
                    {
                        "pipeline_step": "aggregator",
                        "status": "FAILED",
                        "error": str(e),
                    }
                )
                self.mlflow_client.end_run(status="FAILED")
            except Exception as mlflow_err:
                log.warning("Failed to log to MLflow (non-blocking)", error=str(mlflow_err))
            return 1

        except TrainerError as e:
            log.error("Training failed (HARD STOP)", error=str(e))
            try:
                self.mlflow_client.log_params(
                    {
                        "pipeline_step": "trainer",
                        "status": "FAILED",
                        "error": str(e),
                    }
                )
                self.mlflow_client.end_run(status="FAILED")
            except Exception as mlflow_err:
                log.warning("Failed to log to MLflow (non-blocking)", error=str(mlflow_err))
            return 1

        except Exception as e:
            log.error("Unexpected error (HARD STOP)", error=str(e), exc_info=True)
            try:
                self.mlflow_client.log_params(
                    {
                        "pipeline_step": "unknown",
                        "status": "FAILED",
                        "error": str(e),
                    }
                )
                self.mlflow_client.end_run(status="FAILED")
            except Exception as mlflow_err:
                log.warning("Failed to log error to MLflow (non-blocking)", error=str(mlflow_err))
            return 1


def main():
    """Entrypoint for orchestrator."""
    setup_logging(environment="production")

    orchestrator = PipelineOrchestrator()
    exit_code = orchestrator.run()

    if exit_code == 0:
        log.info("Pipeline completed successfully")
    else:
        log.error("Pipeline failed", exit_code=exit_code)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
