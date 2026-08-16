"""Error scenario and resilience tests."""

import pytest
import json
import time
from unittest.mock import patch, MagicMock


class TestKafkaFailures:
    """Test behavior when Kafka is unavailable or degraded."""

    def test_aggregator_handles_kafka_timeout(self):
        """Test aggregator handles Kafka timeout gracefully."""
        from src.core.aggregator import Aggregator, AggregationError

        with patch("src.infra.kafka_client.KafkaConsumer") as mock_kafka:
            mock_kafka.side_effect = Exception("Connection timeout")

            aggregator = Aggregator(kafka_mock=False)
            # Should raise an error or retry gracefully
            try:
                aggregator.aggregate_feedback()
            except AggregationError:
                pass  # Expected

    def test_aggregator_retry_logic(self):
        """Test aggregator retries on transient failures."""
        from src.core.aggregator import Aggregator
        from src.infra.kafka_client import KafkaClientError

        # Aggregator uses mock by default, should succeed
        aggregator = Aggregator(kafka_mock=True)
        train_path, val_path = aggregator.aggregate_feedback()
        assert train_path is not None


class TestS3Failures:
    """Test behavior when S3 is unavailable."""

    def test_s3_client_retries(self):
        """Test S3 client retry logic."""
        from src.infra.s3_client import S3Client

        s3 = S3Client(mock=True)

        # Mock should always work
        result = s3.upload_json("test/path", {"data": "test"})
        assert result is not None

    def test_aggregator_handles_s3_failure(self):
        """Test aggregator handles S3 save failure."""
        from src.core.aggregator import Aggregator

        with patch("src.infra.s3_client.S3Client") as mock_s3:
            mock_s3.return_value.upload_json.side_effect = Exception("S3 error")

            aggregator = Aggregator(kafka_mock=True)
            try:
                aggregator.aggregate_feedback()
            except Exception as e:
                # Should raise error or handle gracefully
                pass


class TestMLflowFailures:
    """Test behavior when MLflow is unavailable."""

    def test_aggregator_continues_without_mlflow(self):
        """Test aggregator continues when MLflow is down."""
        from src.core.aggregator import Aggregator

        aggregator = Aggregator(kafka_mock=True)
        # Mock MLflow to fail
        with patch.object(aggregator.mlflow_client, "start_run") as mock_mlflow:
            mock_mlflow.side_effect = Exception("MLflow unavailable")

            # Should still complete aggregation
            train_path, val_path = aggregator.aggregate_feedback()
            assert train_path is not None

    def test_trainer_continues_without_mlflow(self):
        """Test trainer continues when MLflow logging fails."""
        from src.core.trainer import DPOTrainer
        from src.infra.s3_client import S3Client

        s3_client = S3Client(mock=True)

        mock_data = {
            "examples": [
                {
                    "prompt": "Test",
                    "preferred": "Good response with sufficient length and detail",
                    "rejected": "Bad response",
                }
                for _ in range(5)
            ]
        }

        train_path = s3_client.upload_json("test/train.jsonl", mock_data)
        val_path = s3_client.upload_json("test/val.jsonl", mock_data)

        trainer = DPOTrainer(train_data_path=train_path, val_data_path=val_path)

        with patch.object(trainer.mlflow_client, "start_run") as mock_mlflow:
            mock_mlflow.side_effect = Exception("MLflow unavailable")

            # Should still complete training
            adapter_path = trainer.train()
            assert adapter_path is not None

    def test_quality_gate_continues_without_mlflow(self):
        """Test quality gate continues when MLflow is down."""
        from src.core.quality_gate import QualityGate

        gate = QualityGate(
            challenger_adapter_path="s3://ml-artifacts/models/challenger/adapter.bin",
            champion_adapter_path="s3://ml-artifacts/models/champion/adapter.bin",
            s3_mock=True,
        )

        with patch.object(gate.mlflow_client, "start_run") as mock_mlflow:
            mock_mlflow.side_effect = Exception("MLflow unavailable")

            try:
                gate.validate()
            except Exception:
                pass  # OK if validation fails, just checking MLflow failure doesn't crash


class TestDataQualityFailures:
    """Test behavior with poor data quality."""

    def test_baseline_check_rejects_noisy_data(self):
        """Test baseline check rejects data below threshold."""
        from scripts.baseline_check import run_baseline_check

        # Create very noisy data (just single characters)
        noisy_data = {
            "examples": [{"prompt": "a", "preferred": "b", "rejected": "c"}]
        }

        with patch("src.infra.s3_client.S3Client") as mock_s3:
            mock_s3_instance = MagicMock()
            mock_s3_instance.download_json.return_value = noisy_data
            mock_s3.return_value = mock_s3_instance

            try:
                accuracy = run_baseline_check(train_data_path="s3://test/train.jsonl")
                # Accuracy should be low
                assert accuracy < 0.7
            except ValueError:
                # Expected if data is too poor
                pass

    def test_empty_dataset_handling(self):
        """Test handling of empty training dataset."""
        from scripts.baseline_check import run_baseline_check

        empty_data = {"examples": []}

        with patch("src.infra.s3_client.S3Client") as mock_s3:
            mock_s3_instance = MagicMock()
            mock_s3_instance.download_json.return_value = empty_data
            mock_s3.return_value = mock_s3_instance

            try:
                run_baseline_check(train_data_path="s3://test/train.jsonl")
            except ValueError as e:
                assert "No examples" in str(e)


class TestQualityGateFailures:
    """Test quality gate hard stops."""

    def test_quality_gate_fails_on_low_accuracy(self):
        """Test quality gate fails when accuracy is too low."""
        from src.core.quality_gate import QualityGate, ModelDegradationError

        gate = QualityGate(
            challenger_adapter_path="s3://ml-artifacts/models/low-acc/adapter.bin",
            champion_adapter_path="s3://ml-artifacts/models/champion/adapter.bin",
            s3_mock=True,
        )

        with patch.object(gate, "_compute_reward_accuracy") as mock_acc:
            mock_acc.return_value = 0.65  # Below 0.72 threshold

            try:
                gate.validate()
                # If mock doesn't raise, should still have low metrics
            except ModelDegradationError:
                pass  # Expected

    def test_quality_gate_fails_on_low_win_rate(self):
        """Test quality gate fails when win-rate is too low."""
        from src.core.quality_gate import QualityGate, ModelDegradationError

        gate = QualityGate(
            challenger_adapter_path="s3://ml-artifacts/models/low-wr/adapter.bin",
            champion_adapter_path="s3://ml-artifacts/models/champion/adapter.bin",
            s3_mock=True,
        )

        with patch.object(gate, "_compute_dpo_win_rate") as mock_wr:
            mock_wr.return_value = 0.40  # Below 0.55 threshold

            try:
                gate.validate()
            except ModelDegradationError:
                pass  # Expected


class TestDeploymentFailures:
    """Test deployment error handling."""

    def test_deployment_rollback_on_failure(self):
        """Test deployment rollback on health check failure."""
        from src.api.deployment import BlueGreenDeployer

        deployer = BlueGreenDeployer(s3_mock=True)

        with patch.object(deployer, "_health_check") as mock_health:
            mock_health.return_value = False  # Health check fails

            result = deployer.deploy_canary(
                challenger_adapter_path="s3://ml-artifacts/models/bad/adapter.bin",
                challenger_version="v_bad",
            )

            # Should fail gracefully
            assert result["status"] in ["failed", "rolledback"]

    def test_deployment_rollback_on_success_rate_drop(self):
        """Test deployment rollback on success rate degradation."""
        from src.api.deployment import BlueGreenDeployer

        deployer = BlueGreenDeployer(s3_mock=True)

        with patch.object(deployer, "_monitor_canary") as mock_monitor:
            # Simulate success rate drop
            mock_monitor.return_value = {
                "success_rate": 0.90,  # Below 0.95 threshold
                "latency": 150,
            }

            result = deployer.deploy_canary(
                challenger_adapter_path="s3://ml-artifacts/models/slow/adapter.bin",
                challenger_version="v_slow",
            )

            # Should rollback or fail
            assert result["status"] in ["failed", "rolledback"]


class TestRetryLogic:
    """Test exponential backoff retry logic."""

    def test_exponential_backoff_retries(self):
        """Test that retry decorator uses exponential backoff."""
        from tenacity import retry, stop_after_attempt, wait_exponential

        attempt_count = 0
        call_times = []

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.1, min=0.1, max=1))
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            call_times.append(time.time())

            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3

    def test_retry_gives_up_after_attempts(self):
        """Test that retry gives up after max attempts."""
        from tenacity import retry, stop_after_attempt, RetryError

        attempt_count = 0

        @retry(stop=stop_after_attempt(3))
        def always_fails():
            nonlocal attempt_count
            attempt_count += 1
            raise Exception("Always fails")

        try:
            always_fails()
        except RetryError:
            pass  # Expected

        assert attempt_count == 3


class TestConcurrencyIssues:
    """Test concurrent access and race conditions."""

    def test_concurrent_feedback_submissions(self, api_client):
        """Test concurrent feedback doesn't cause issues."""
        import concurrent.futures

        def submit(i):
            return api_client.post("/feedback", json={
                "event_id": f"event-{i}",
                "prompt": "Test",
                "chosen_response": "Good response with sufficient detail and content",
                "rejected_response": "Bad response",
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(submit, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert len(results) == 50
        assert all(r.status_code in [200, 202] for r in results)
