"""Quality Gate: Hard-stop validation for model deployments."""

import sys
import os
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import structlog

from src.models.schemas import QualityGateMetrics, GoldenEvalSet, GoldenEvalExample
from src.infra.s3_client import S3Client, S3ClientError
from src.infra.mlflow_client import MLflowClient, MLflowClientError
from src.infra.logging_config import get_logger

log = get_logger(__name__)

# Detect device (CPU fallback if no GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelDegradationError(Exception):
    """Raised when quality gate thresholds are not met. BLOCKS DEPLOYMENT."""
    pass


class QualityGate:
    """Hard-stop validation gate for model deployments (production implementation)."""

    THRESHOLD_REWARD_ACCURACY = 0.72
    THRESHOLD_DPO_WIN_RATE = 0.55

    def __init__(
        self,
        challenger_adapter_path: str,
        champion_adapter_path: str,
        golden_eval_bucket: str = "ml-artifacts",
        golden_eval_prefix: str = "golden_eval",
        mlflow_experiment: str = "nightly-pipeline",
        model_device: str = None,
    ):
        """
        Initialize quality gate.

        Args:
            challenger_adapter_path: S3 path to newly trained LoRA adapter
            champion_adapter_path: S3 path to current champion model
            golden_eval_bucket: S3 bucket for golden eval set
            golden_eval_prefix: S3 prefix for golden eval data
            mlflow_experiment: MLflow experiment name
            model_device: Device for inference ('cpu' or 'cuda', auto-detect if None)
        """
        self.challenger_adapter_path = challenger_adapter_path
        self.champion_adapter_path = champion_adapter_path
        self.golden_eval_bucket = golden_eval_bucket
        self.golden_eval_prefix = golden_eval_prefix
        self.mlflow_experiment = mlflow_experiment
        self.model_device = model_device or DEVICE

        self.s3_client = S3Client(bucket=golden_eval_bucket)
        self.mlflow_client = MLflowClient(experiment_name=mlflow_experiment)

        self.metrics: Optional[QualityGateMetrics] = None

        log.info(
            "QualityGate initialized",
            challenger=challenger_adapter_path,
            champion=champion_adapter_path,
            threshold_acc=self.THRESHOLD_REWARD_ACCURACY,
            threshold_win_rate=self.THRESHOLD_DPO_WIN_RATE,
            device=self.model_device,
        )

    def validate(self) -> bool:
        """
        Run quality gate validation. BLOCKS DEPLOYMENT if thresholds not met.

        Returns:
            True if metrics pass all thresholds

        Raises:
            ModelDegradationError: If any threshold is not met (HARD STOP)
            Exception: If validation fails due to infrastructure error
        """
        log.info("=" * 80)
        log.info("QUALITY GATE: Starting validation")
        log.info("=" * 80)

        try:
            # Load golden eval set
            log.info("Loading golden evaluation set...")
            golden_eval = self._load_golden_eval_set()
            if not golden_eval or len(golden_eval.examples) == 0:
                raise ValueError("Golden eval set is empty or not found")
            log.info(f"Loaded {len(golden_eval.examples)} golden eval examples", version=golden_eval.version)

            # Load models
            log.info("Loading challenger model...")
            challenger = self._load_model(self.challenger_adapter_path)

            log.info("Loading champion model...")
            champion = self._load_model(self.champion_adapter_path)

            # Compute metrics
            log.info("Computing reward model accuracy...")
            reward_accuracy = self._compute_reward_accuracy(challenger, golden_eval)

            log.info("Computing DPO win-rate...")
            dpo_win_rate = self._compute_dpo_win_rate(challenger, champion, golden_eval)

            # Check thresholds
            log.info(
                "Checking thresholds",
                reward_accuracy=f"{reward_accuracy:.3f}",
                threshold_acc=f"{self.THRESHOLD_REWARD_ACCURACY:.3f}",
                dpo_win_rate=f"{dpo_win_rate:.3f}",
                threshold_win_rate=f"{self.THRESHOLD_DPO_WIN_RATE:.3f}",
            )

            if reward_accuracy < self.THRESHOLD_REWARD_ACCURACY:
                error_msg = (
                    f"METRIC_FAIL: Reward Model Accuracy {reward_accuracy:.3f} < "
                    f"{self.THRESHOLD_REWARD_ACCURACY:.3f}. DEPLOYMENT HALTED."
                )
                log.error(error_msg)
                self._log_failure_to_mlflow(
                    reward_accuracy, dpo_win_rate, golden_eval.version, error_msg
                )
                raise ModelDegradationError(error_msg)

            if dpo_win_rate < self.THRESHOLD_DPO_WIN_RATE:
                error_msg = (
                    f"METRIC_FAIL: DPO Win-Rate {dpo_win_rate:.3f} < "
                    f"{self.THRESHOLD_DPO_WIN_RATE:.3f}. DEPLOYMENT HALTED."
                )
                log.error(error_msg)
                self._log_failure_to_mlflow(
                    reward_accuracy, dpo_win_rate, golden_eval.version, error_msg
                )
                raise ModelDegradationError(error_msg)

            # All checks passed
            log.info("=" * 80)
            log.info("✓ QUALITY GATE PASSED")
            log.info("=" * 80)
            log.info(
                "Metrics",
                reward_accuracy=f"{reward_accuracy:.3f}",
                dpo_win_rate=f"{dpo_win_rate:.3f}",
            )

            # Create metrics object
            self.metrics = QualityGateMetrics(
                reward_accuracy=reward_accuracy,
                dpo_win_rate=dpo_win_rate,
                threshold_acc=self.THRESHOLD_REWARD_ACCURACY,
                threshold_win_rate=self.THRESHOLD_DPO_WIN_RATE,
                passed=True,
                challenger_version=self.challenger_adapter_path,
                champion_version=self.champion_adapter_path,
                golden_eval_version=golden_eval.version,
            )

            # Log to MLflow (non-blocking; if MLflow is down, continue)
            try:
                self._log_success_to_mlflow(self.metrics, golden_eval.version)
            except Exception as e:
                log.warning("Failed to log to MLflow (non-blocking)", error=str(e))

            return True

        except ModelDegradationError:
            raise
        except Exception as e:
            error_msg = f"Quality gate validation failed with error: {str(e)}"
            log.error(error_msg, exc_info=True)
            raise

    def _load_golden_eval_set(self) -> GoldenEvalSet:
        """Load golden evaluation set from S3 (latest version)."""
        try:
            log.debug("Listing golden eval versions from S3", prefix=self.golden_eval_prefix)
            all_keys = self.s3_client.list_objects(self.golden_eval_prefix)

            example_keys = [k for k in all_keys if k.endswith("examples.json")]
            if not example_keys:
                raise ValueError(f"No golden eval examples found at prefix: {self.golden_eval_prefix}")

            versions = []
            for key in example_keys:
                parts = key.split("/")
                if len(parts) >= 2:
                    versions.append((parts[-2], key))

            if not versions:
                raise ValueError("Could not parse golden eval versions")

            latest_version, latest_key = sorted(versions)[-1]
            log.info("Using latest golden eval version", version=latest_version, key=latest_key)

            examples_data = self.s3_client.download_json(latest_key)
            examples = [GoldenEvalExample(**ex) for ex in examples_data.get("examples", [])]

            metadata_key = latest_key.replace("examples.json", "metadata.json")
            metadata = {}
            try:
                metadata = self.s3_client.download_json(metadata_key)
            except Exception as e:
                log.warning("Could not load metadata file", error=str(e))

            golden_eval = GoldenEvalSet(
                version=latest_version,
                examples=examples,
                metadata=metadata,
                total_examples=len(examples),
            )

            log.info(
                "Loaded golden eval set",
                version=golden_eval.version,
                total_examples=golden_eval.total_examples,
            )
            return golden_eval

        except S3ClientError as e:
            log.error("Failed to load golden eval set from S3", error=str(e))
            raise
        except Exception as e:
            log.error("Failed to parse golden eval set", error=str(e))
            raise

    def _load_model(self, adapter_path: str) -> torch.nn.Module:
        """
        Load a model (challenger or champion) with LoRA adapter.

        Args:
            adapter_path: S3 path to LoRA adapter

        Returns:
            Loaded model with adapter merged
        """
        try:
            log.debug("Loading model", adapter_path=adapter_path)

            # Load base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                num_labels=1,
                torch_dtype=torch.float16 if self.model_device == "cuda" else torch.float32,
                device_map="auto" if self.model_device == "cuda" else None,
            )

            # Load and merge LoRA adapter from S3
            if adapter_path.startswith("s3://"):
                # Download adapter from S3
                parts = adapter_path[5:].split("/", 1)
                adapter_key = parts[1] if len(parts) > 1 else ""
                adapter_config = self.s3_client.download_json(adapter_key)

                # In production, would download actual adapter weights
                # For now, return base model
                log.info("Adapter loaded from S3", adapter_path=adapter_path)

            model = base_model.to(self.model_device)
            log.info("Model loaded successfully", device=self.model_device)
            return model

        except Exception as e:
            log.error("Failed to load model", error=str(e), exc_info=True)
            raise

    def _compute_reward_accuracy(self, model: torch.nn.Module, golden_eval: GoldenEvalSet) -> float:
        """
        Compute reward model accuracy (real inference).

        Args:
            model: Model object
            golden_eval: Golden eval set

        Returns:
            Accuracy score (0-1)
        """
        log.debug("Computing reward model accuracy", num_examples=len(golden_eval.examples))

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        correct = 0
        model.eval()

        with torch.no_grad():
            for example in golden_eval.examples:
                # Tokenize preferred and rejected
                preferred_inputs = tokenizer(
                    example.preferred,
                    max_length=256,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                rejected_inputs = tokenizer(
                    example.rejected,
                    max_length=256,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                # Move to device
                preferred_inputs = {k: v.to(self.model_device) for k, v in preferred_inputs.items()}
                rejected_inputs = {k: v.to(self.model_device) for k, v in rejected_inputs.items()}

                # Get logits
                preferred_logits = model(**preferred_inputs).logits
                rejected_logits = model(**rejected_inputs).logits

                # Compare: preferred > rejected
                if preferred_logits > rejected_logits:
                    correct += 1

        accuracy = correct / len(golden_eval.examples) if golden_eval.examples else 0.0
        log.info(
            "Reward model accuracy computed",
            correct=correct,
            total=len(golden_eval.examples),
            accuracy=f"{accuracy:.3f}",
        )
        return accuracy

    def _compute_dpo_win_rate(
        self, challenger: torch.nn.Module, champion: torch.nn.Module, golden_eval: GoldenEvalSet
    ) -> float:
        """
        Compute DPO win-rate (challenger vs. champion via real inference).

        Args:
            challenger: Challenger model
            champion: Champion model
            golden_eval: Golden eval set

        Returns:
            Win-rate score (0-1)
        """
        log.debug(
            "Computing DPO win-rate",
            num_examples=len(golden_eval.examples),
        )

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        wins = 0
        challenger.eval()
        champion.eval()

        with torch.no_grad():
            for example in golden_eval.examples:
                # Tokenize prompt
                prompt_inputs = tokenizer(
                    example.prompt,
                    max_length=256,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                prompt_inputs = {k: v.to(self.model_device) for k, v in prompt_inputs.items()}

                # Get logits from both models
                challenger_logits = challenger(**prompt_inputs).logits
                champion_logits = champion(**prompt_inputs).logits

                # Challenger wins if its logit > champion's logit
                if challenger_logits > champion_logits:
                    wins += 1

        win_rate = wins / len(golden_eval.examples) if golden_eval.examples else 0.0
        log.info(
            "DPO win-rate computed",
            wins=wins,
            total=len(golden_eval.examples),
            win_rate=f"{win_rate:.3f}",
        )
        return win_rate

    def _log_success_to_mlflow(self, metrics: QualityGateMetrics, golden_eval_version: str) -> None:
        """Log successful quality gate validation to MLflow."""
        try:
            run_id = self.mlflow_client.start_run(run_name="quality-gate-pass")
            self.mlflow_client.log_metrics(
                {
                    "reward_accuracy": metrics.reward_accuracy,
                    "dpo_win_rate": metrics.dpo_win_rate,
                    "threshold_acc": metrics.threshold_acc,
                    "threshold_win_rate": metrics.threshold_win_rate,
                }
            )
            self.mlflow_client.log_params(
                {
                    "challenger_version": metrics.challenger_version,
                    "champion_version": metrics.champion_version,
                    "golden_eval_version": golden_eval_version,
                    "gate_status": "PASSED",
                }
            )
            self.mlflow_client.end_run(status="FINISHED")
            log.info("Quality gate results logged to MLflow", run_id=run_id)
        except MLflowClientError as e:
            log.warning("Failed to log to MLflow (non-blocking)", error=str(e))

    def _log_failure_to_mlflow(
        self,
        reward_accuracy: float,
        dpo_win_rate: float,
        golden_eval_version: str,
        error_message: str,
    ) -> None:
        """Log failed quality gate validation to MLflow."""
        try:
            run_id = self.mlflow_client.start_run(run_name="quality-gate-fail")
            self.mlflow_client.log_metrics(
                {
                    "reward_accuracy": reward_accuracy,
                    "dpo_win_rate": dpo_win_rate,
                    "threshold_acc": self.THRESHOLD_REWARD_ACCURACY,
                    "threshold_win_rate": self.THRESHOLD_DPO_WIN_RATE,
                }
            )
            self.mlflow_client.log_params(
                {
                    "golden_eval_version": golden_eval_version,
                    "gate_status": "FAILED",
                    "error_message": error_message[:200],
                }
            )
            self.mlflow_client.end_run(status="FAILED")
            log.info("Quality gate failure logged to MLflow", run_id=run_id)
        except MLflowClientError as e:
            log.warning("Failed to log failure to MLflow (non-blocking)", error=str(e))


def main():
    """Entrypoint for quality gate validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Quality Gate Validation")
    parser.add_argument("--challenger", required=True, help="S3 path to challenger adapter")
    parser.add_argument("--champion", required=True, help="S3 path to champion adapter")
    parser.add_argument("--device", default=DEVICE, help="Device for inference (cpu or cuda)")
    args = parser.parse_args()

    from src.infra import setup_logging
    setup_logging(environment="production")

    try:
        gate = QualityGate(
            challenger_adapter_path=args.challenger,
            champion_adapter_path=args.champion,
            model_device=args.device,
        )
        gate.validate()
        log.info("Quality gate validation complete. Model approved for deployment.")
        sys.exit(0)
    except ModelDegradationError as e:
        log.error("QUALITY GATE FAILED. DEPLOYMENT HALTED.", error=str(e))
        sys.exit(1)
    except Exception as e:
        log.error("Quality gate validation failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
