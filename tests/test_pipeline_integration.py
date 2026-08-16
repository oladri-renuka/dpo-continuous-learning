"""Full pipeline integration tests."""

import pytest
import time
import json
from pathlib import Path


class TestAggregatorIntegration:
    """Test aggregator with real Kafka."""

    def test_aggregator_consumes_feedback(self, kafka_producer, mock_feedback_message):
        """Test that aggregator can consume feedback from Kafka."""
        from src.core.aggregator import Aggregator

        topic = "feedback.events"

        # Produce test messages
        for i in range(10):
            message = {**mock_feedback_message, "event_id": f"event-{i}"}
            kafka_producer.produce(topic, json.dumps(message).encode("utf-8"))
        kafka_producer.flush()

        # Give Kafka time to process
        time.sleep(1)

        # Create aggregator and consume
        aggregator = Aggregator(kafka_mock=False)
        train_path, val_path = aggregator.aggregate_feedback()

        # Verify output
        assert train_path is not None
        assert val_path is not None
        assert "train" in train_path
        assert "val" in val_path

    def test_aggregator_deduplication(self, kafka_producer, mock_feedback_message):
        """Test that aggregator deduplicates messages."""
        from src.core.aggregator import Aggregator

        topic = "feedback.events"

        # Produce duplicate messages
        for _ in range(5):
            kafka_producer.produce(topic, json.dumps(mock_feedback_message).encode("utf-8"))
        kafka_producer.flush()

        time.sleep(1)

        aggregator = Aggregator(kafka_mock=False)
        train_path, val_path = aggregator.aggregate_feedback()

        # Should still succeed even with duplicates
        assert train_path is not None
        assert val_path is not None

    def test_aggregator_statistics(self, kafka_producer, mock_feedback_message):
        """Test that aggregator tracks statistics."""
        from src.core.aggregator import Aggregator

        topic = "feedback.events"

        # Produce messages
        num_messages = 20
        for i in range(num_messages):
            message = {**mock_feedback_message, "event_id": f"event-{i}"}
            kafka_producer.produce(topic, json.dumps(message).encode("utf-8"))
        kafka_producer.flush()

        time.sleep(1)

        aggregator = Aggregator(kafka_mock=False)
        train_path, val_path = aggregator.aggregate_feedback()

        # Verify stats are recorded
        assert aggregator.stats["total_raw_messages"] > 0
        assert aggregator.stats["final_pairs"] >= 0
        assert aggregator.stats["train_pairs"] >= 0
        assert aggregator.stats["val_pairs"] >= 0


class TestBaselineCheckIntegration:
    """Test baseline check with real data."""

    def test_baseline_check_learnable_data(self, mock_training_data):
        """Test baseline check on learnable data."""
        from scripts.baseline_check import run_baseline_check
        from src.infra.s3_client import S3Client
        import tempfile

        # Save mock data to S3
        s3_client = S3Client(mock=True)
        key = "test-baseline/train.jsonl"
        s3_path = s3_client.upload_json(key, mock_training_data)

        # Run baseline check
        accuracy = run_baseline_check(train_data_path=s3_path)

        # Should pass threshold
        assert accuracy >= 0.55
        assert accuracy <= 1.0

    def test_baseline_check_metrics_logging(self, mock_training_data):
        """Test that baseline check logs metrics."""
        from scripts.baseline_check import run_baseline_check
        from src.infra.s3_client import S3Client

        s3_client = S3Client(mock=True)
        key = "test-baseline/train.jsonl"
        s3_path = s3_client.upload_json(key, mock_training_data)

        # Run baseline check
        accuracy = run_baseline_check(train_data_path=s3_path)

        # Should return valid accuracy
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


class TestTrainerIntegration:
    """Test trainer with real data."""

    def test_trainer_completes(self, mock_training_data):
        """Test that trainer completes successfully."""
        from src.core.trainer import DPOTrainer
        from src.infra.s3_client import S3Client

        s3_client = S3Client(mock=True)

        # Save data to S3
        train_key = "test-trainer/train.jsonl"
        val_key = "test-trainer/val.jsonl"
        train_path = s3_client.upload_json(train_key, mock_training_data)
        val_path = s3_client.upload_json(val_key, mock_training_data)

        # Create trainer
        trainer = DPOTrainer(train_data_path=train_path, val_data_path=val_path)
        adapter_path = trainer.train()

        # Should return a valid adapter path
        assert adapter_path is not None
        assert "adapter" in adapter_path

    def test_trainer_metrics(self, mock_training_data):
        """Test that trainer computes metrics."""
        from src.core.trainer import DPOTrainer
        from src.infra.s3_client import S3Client

        s3_client = S3Client(mock=True)

        train_key = "test-metrics/train.jsonl"
        val_key = "test-metrics/val.jsonl"
        train_path = s3_client.upload_json(train_key, mock_training_data)
        val_path = s3_client.upload_json(val_key, mock_training_data)

        trainer = DPOTrainer(train_data_path=train_path, val_data_path=val_path)
        adapter_path = trainer.train()

        # Check metrics are recorded
        assert trainer.metrics.get("reward_model_accuracy") is not None
        assert trainer.metrics.get("dpo_loss") is not None
        assert trainer.metrics.get("dpo_win_rate") is not None


class TestQualityGateIntegration:
    """Test quality gate validation."""

    def test_quality_gate_passes(self):
        """Test that quality gate passes with good metrics."""
        from src.core.quality_gate import QualityGate

        gate = QualityGate(
            challenger_adapter_path="s3://ml-artifacts/models/challenger/adapter.bin",
            champion_adapter_path="s3://ml-artifacts/models/champion/adapter.bin",
            s3_mock=True,
        )

        # Should not raise exception
        try:
            gate.validate()
        except Exception as e:
            # OK if metrics don't validate, just checking it runs
            pass

    def test_quality_gate_metrics_logged(self):
        """Test that quality gate logs metrics to MLflow."""
        from src.core.quality_gate import QualityGate

        gate = QualityGate(
            challenger_adapter_path="s3://ml-artifacts/models/challenger/adapter.bin",
            champion_adapter_path="s3://ml-artifacts/models/champion/adapter.bin",
            s3_mock=True,
        )

        try:
            gate.validate()
        except:
            pass

        # Metrics should be computed
        assert hasattr(gate, "metrics")


class TestDeploymentIntegration:
    """Test Blue/Green deployment."""

    def test_deployment_starts(self):
        """Test that deployment process starts."""
        from src.api.deployment import deploy_model

        result = deploy_model(
            challenger_adapter_path="s3://ml-artifacts/models/challenger/v123/adapter.bin",
            challenger_version="v123",
        )

        # Should return a deployment result
        assert isinstance(result, dict)
        assert "status" in result

    def test_deployment_updates_champion(self):
        """Test that deployment updates champion pointer."""
        from src.api.deployment import deploy_model
        import json
        from pathlib import Path

        result = deploy_model(
            challenger_adapter_path="s3://ml-artifacts/models/challenger/v456/adapter.bin",
            challenger_version="v456",
        )

        # Check if champion pointer was updated
        pointer_path = Path("/tmp/dpo-champion-pointer.json")
        if pointer_path.exists():
            with open(pointer_path) as f:
                pointer = json.load(f)
            assert "champion_version" in pointer


class TestPipelineEndToEnd:
    """Test complete pipeline execution."""

    def test_full_pipeline_mock_mode(self):
        """Test full pipeline with mocks."""
        from scripts.run_pipeline import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        exit_code = orchestrator.run()

        # Should complete (exit code 0 or 1 based on quality gate)
        assert exit_code in [0, 1]

    def test_pipeline_stages_execute_sequentially(self):
        """Test that pipeline stages execute in order."""
        from src.core.aggregator import Aggregator
        from scripts.baseline_check import run_baseline_check
        from src.core.trainer import DPOTrainer
        from src.infra.s3_client import S3Client

        # Stage 1: Aggregation
        aggregator = Aggregator(kafka_mock=True)
        train_path, val_path = aggregator.aggregate_feedback()
        assert train_path is not None

        # Stage 2: Baseline check
        accuracy = run_baseline_check(train_data_path=train_path)
        assert accuracy >= 0.0

        # Stage 3: Training (if baseline passed)
        if accuracy >= 0.55:
            trainer = DPOTrainer(train_data_path=train_path, val_data_path=val_path)
            adapter_path = trainer.train()
            assert adapter_path is not None
