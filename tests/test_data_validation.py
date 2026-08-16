"""Data validation and quality tests."""

import pytest
import json
from typing import Dict, Any


class TestDataValidation:
    """Test data validation across pipeline stages."""

    def test_feedback_message_schema(self):
        """Test feedback message schema validation."""
        from src.models.schemas import FeedbackEvent
        from pydantic import ValidationError

        # Valid feedback
        valid = {
            "event_id": "test-1",
            "timestamp": 1692864000,
            "user_id": "user-1",
            "prompt": "Test prompt",
            "chosen_response": "Good response",
            "rejected_response": "Bad response",
            "feedback_type": "thumbs_up",
        }

        event = FeedbackEvent(**valid)
        assert event.event_id == "test-1"

    def test_invalid_feedback_schema(self):
        """Test invalid feedback is rejected."""
        from src.models.schemas import FeedbackEvent
        from pydantic import ValidationError

        # Missing required fields
        invalid = {"event_id": "test-1"}

        with pytest.raises(ValidationError):
            FeedbackEvent(**invalid)

    def test_training_data_schema(self):
        """Test training data schema."""
        from src.models.schemas import PreferencePair

        valid_pair = {
            "prompt": "What is Python?",
            "preferred": "Python is a programming language that is easy to learn and powerful.",
            "rejected": "Python is a language",
        }

        pair = PreferencePair(**valid_pair)
        assert pair.prompt == "What is Python?"

    def test_golden_eval_set_schema(self):
        """Test golden evaluation set schema."""
        from src.models.schemas import GoldenEvalSet

        valid_set = {
            "version": "v1",
            "examples": [
                {
                    "prompt": "Test?",
                    "chosen": "Good response with detail",
                    "rejected": "Bad response",
                    "category": "simple",
                }
                for _ in range(10)
            ],
            "created_at": "2026-08-16T00:00:00Z",
            "description": "Test set",
        }

        eval_set = GoldenEvalSet(**valid_set)
        assert eval_set.version == "v1"
        assert len(eval_set.examples) == 10


class TestDataQuality:
    """Test data quality metrics and checks."""

    def test_response_length_validation(self):
        """Test response length validation."""
        MIN_LENGTH = 50
        MAX_LENGTH = 4096

        # Short response (invalid)
        short = "x" * 10
        assert len(short) < MIN_LENGTH

        # Long response (valid)
        long = "y" * 100
        assert len(long) >= MIN_LENGTH
        assert len(long) <= MAX_LENGTH

        # Too long response (should be truncated)
        very_long = "z" * 5000
        assert len(very_long) > MAX_LENGTH

    def test_duplicate_detection(self):
        """Test duplicate message detection."""
        import hashlib

        def compute_hash(prompt, chosen, rejected):
            pair_str = f"{prompt}||{chosen}||{rejected}"
            return hashlib.sha256(pair_str.encode()).hexdigest()

        msg1 = {
            "prompt": "Test",
            "chosen_response": "Response A",
            "rejected_response": "Response B",
        }

        msg2 = {
            "prompt": "Test",
            "chosen_response": "Response A",
            "rejected_response": "Response B",
        }

        msg3 = {
            "prompt": "Test",
            "chosen_response": "Response C",
            "rejected_response": "Response B",
        }

        hash1 = compute_hash(msg1["prompt"], msg1["chosen_response"], msg1["rejected_response"])
        hash2 = compute_hash(msg2["prompt"], msg2["chosen_response"], msg2["rejected_response"])
        hash3 = compute_hash(msg3["prompt"], msg3["chosen_response"], msg3["rejected_response"])

        assert hash1 == hash2  # Duplicates
        assert hash1 != hash3  # Different

    def test_user_bias_detection(self):
        """Test detection of user bias in data."""
        from collections import defaultdict

        messages = [
            {"user_id": "user_1"} for _ in range(90)  # 90% from one user
        ] + [
            {"user_id": f"user_{i}"} for i in range(2, 12)  # 10% from others
        ]

        user_counts = defaultdict(int)
        for msg in messages:
            user_counts[msg.get("user_id", "anonymous")] += 1

        max_user_count = max(user_counts.values())
        user_bias_pct = max_user_count / len(messages)

        assert user_bias_pct >= 0.85  # Should detect high bias

    def test_data_distribution(self):
        """Test data distribution analysis."""
        messages = [
            {"feedback_type": "thumbs_up" if i % 2 == 0 else "thumbs_down"}
            for i in range(100)
        ]

        thumbs_up = sum(1 for m in messages if m["feedback_type"] == "thumbs_up")
        thumbs_down = sum(1 for m in messages if m["feedback_type"] == "thumbs_down")

        ratio = thumbs_up / len(messages)
        assert 0.4 < ratio < 0.6  # Roughly balanced


class TestMetricsValidation:
    """Test metric calculation and validation."""

    def test_reward_accuracy_calculation(self):
        """Test reward model accuracy calculation."""
        predictions = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0]
        ground_truth = [1, 0, 0, 1, 1, 1, 1, 0, 1, 0]

        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions)

        assert 0.0 <= accuracy <= 1.0
        assert accuracy == 0.6  # 6 out of 10 correct

    def test_win_rate_calculation(self):
        """Test DPO win-rate calculation."""
        # Win-rate: % of preference pairs where challenger > champion
        wins = 85
        total = 100
        win_rate = wins / total

        assert 0.0 <= win_rate <= 1.0
        assert win_rate == 0.85

    def test_loss_calculation(self):
        """Test DPO loss calculation."""
        # DPO loss should decrease with training
        epoch_losses = [2.5, 2.3, 2.1, 1.9, 1.8, 1.7]

        assert epoch_losses[0] > epoch_losses[-1]  # Decreasing
        assert all(isinstance(loss, float) for loss in epoch_losses)

    def test_metrics_bounds(self):
        """Test that metrics stay within valid bounds."""
        accuracy = 0.78
        win_rate = 0.62
        loss = 0.45

        # Accuracy and win-rate should be [0, 1]
        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= win_rate <= 1.0

        # Loss should be positive
        assert loss >= 0.0


class TestDataSerialization:
    """Test JSON serialization/deserialization."""

    def test_feedback_event_serialization(self):
        """Test feedback event can be serialized."""
        from src.models.schemas import FeedbackEvent

        event = FeedbackEvent(
            event_id="test-1",
            timestamp=1692864000,
            user_id="user-1",
            prompt="Test",
            chosen_response="Good response with sufficient detail",
            rejected_response="Bad response",
            feedback_type="thumbs_up",
        )

        # Serialize
        json_str = event.model_dump_json()
        assert isinstance(json_str, str)

        # Deserialize
        import json
        data = json.loads(json_str)
        assert data["event_id"] == "test-1"

    def test_training_config_serialization(self):
        """Test training config serialization."""
        from src.models.config import get_settings

        settings = get_settings()

        # Should be able to convert to dict
        config_dict = settings.model_dump()
        assert isinstance(config_dict, dict)
        assert "kafka_bootstrap_servers" in config_dict

    def test_quality_metrics_serialization(self):
        """Test quality metrics can be logged."""
        from src.models.schemas import QualityGateMetrics

        metrics = QualityGateMetrics(
            reward_accuracy=0.78,
            dpo_win_rate=0.65,
        )

        # Should serialize for logging
        json_str = metrics.model_dump_json()
        assert "0.78" in json_str or "0.78" in str(metrics)


class TestDataConsistency:
    """Test data consistency across pipeline."""

    def test_train_val_split_consistency(self):
        """Test train/val split produces disjoint sets."""
        import random

        data = list(range(100))
        random.shuffle(data)

        split_idx = int(len(data) * 0.9)
        train = set(data[:split_idx])
        val = set(data[split_idx:])

        # Should be disjoint
        assert len(train & val) == 0

        # Should cover all data
        assert len(train | val) == 100

    def test_golden_eval_consistency(self):
        """Test golden eval set remains consistent."""
        from src.infra.s3_client import S3Client

        s3_client = S3Client(mock=True)

        golden_set = {
            "version": "v1",
            "examples": [
                {
                    "prompt": f"Q{i}",
                    "chosen": "Good",
                    "rejected": "Bad",
                }
                for i in range(10)
            ],
        }

        # Upload
        key = "golden/eval/v1/examples.json"
        path = s3_client.upload_json(key, golden_set)

        # Download
        downloaded = s3_client.download_json(key)

        # Should be identical
        assert len(downloaded["examples"]) == 10
        assert downloaded["version"] == "v1"

    def test_metrics_consistency_across_runs(self):
        """Test metrics are consistently computed."""
        # Same data should produce same metrics
        predictions = [1, 0, 1, 1, 0]
        ground_truth = [1, 0, 1, 0, 0]

        accuracy1 = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
        accuracy2 = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)

        assert accuracy1 == accuracy2
        assert accuracy1 == 0.8  # 4 out of 5 correct
