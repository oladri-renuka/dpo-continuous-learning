"""Nightly Kafka aggregation: consume feedback → create training datasets."""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import structlog

from src.infra import get_logger
from src.infra.kafka_client import KafkaConsumer, KafkaClientError
from src.infra.s3_client import S3Client, S3ClientError
from src.infra.mlflow_client import MLflowClient, MLflowClientError
from src.models.config import get_settings

log = get_logger(__name__)


class AggregationError(Exception):
    """Custom exception for aggregation failures."""

    pass


class Aggregator:
    """
    Nightly Kafka aggregation pipeline.

    Consumes 24h of user feedback from Kafka, deduplicates, filters, and
    creates train/val datasets for the DPO training pipeline.
    """

    def __init__(self):
        """Initialize aggregator with config and clients (production only)."""
        self.settings = get_settings()
        self.kafka_consumer = KafkaConsumer(
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            topic=self.settings.kafka_topic_feedback,
        )
        self.s3_client = S3Client(bucket=self.settings.s3_bucket)
        self.mlflow_client = MLflowClient()

        # Aggregation stats
        self.stats = {
            "total_raw_messages": 0,
            "valid_messages": 0,
            "duplicate_messages": 0,
            "filtered_messages": 0,
            "final_pairs": 0,
            "train_pairs": 0,
            "val_pairs": 0,
            "dedup_rate": 0.0,
            "filter_rate": 0.0,
        }

    def aggregate_feedback(self) -> Tuple[str, str]:
        """
        Run the complete aggregation pipeline.

        Returns:
            Tuple of (train_s3_path, val_s3_path)

        Raises:
            AggregationError: If aggregation fails
        """
        log.info("=" * 80)
        log.info("AGGREGATOR: Starting nightly feedback aggregation")
        log.info("=" * 80)

        try:
            # Step 1: Consume from Kafka
            log.info("STEP 1: Consuming feedback from Kafka...")
            raw_messages = self._consume_kafka()

            # Step 2: Deduplicate
            log.info("STEP 2: Deduplicating messages...")
            deduplicated = self._dedup(raw_messages)

            # Step 3: Filter
            log.info("STEP 3: Filtering low-quality examples...")
            filtered = self._filter(deduplicated)

            # Step 4: Train/Val split
            log.info("STEP 4: Splitting into train/val sets...")
            train_pairs, val_pairs = self._train_val_split(filtered)

            # Step 5: Save to S3
            log.info("STEP 5: Saving to S3...")
            train_path, val_path = self._save_to_s3(train_pairs, val_pairs)

            # Log to MLflow (non-blocking; if MLflow is down, continue)
            try:
                self._log_to_mlflow()
            except Exception as e:
                log.warning("Failed to log to MLflow (non-blocking)", error=str(e))

            log.info("=" * 80)
            log.info("✓ AGGREGATION COMPLETE")
            log.info("=" * 80)
            log.info(
                "Aggregation stats",
                total_raw=self.stats["total_raw_messages"],
                valid=self.stats["valid_messages"],
                duplicates=self.stats["duplicate_messages"],
                filtered=self.stats["filtered_messages"],
                final_pairs=self.stats["final_pairs"],
                train_pairs=self.stats["train_pairs"],
                val_pairs=self.stats["val_pairs"],
                dedup_rate=f"{self.stats['dedup_rate']:.1%}",
                filter_rate=f"{self.stats['filter_rate']:.1%}",
            )

            return train_path, val_path

        except Exception as e:
            log.error("Aggregation failed", error=str(e), exc_info=True)
            raise AggregationError(f"Aggregation failed: {str(e)}") from e

        finally:
            self.kafka_consumer.close()

    def _consume_kafka(self) -> List[Dict[str, Any]]:
        """
        Consume feedback messages from Kafka.

        Returns:
            List of raw messages
        """
        try:
            messages = self.kafka_consumer.consume_batch(num_messages=10000, timeout_ms=30000)
            self.stats["total_raw_messages"] = len(messages)
            log.info(
                "Consumed from Kafka",
                topic=self.settings.kafka_topic_feedback,
                messages=len(messages),
            )
            return messages
        except KafkaClientError as e:
            log.error("Failed to consume from Kafka", error=str(e))
            raise AggregationError(f"Kafka consumption failed: {str(e)}") from e

    def _dedup(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate messages by (prompt, response_pair) hash.

        Args:
            messages: Raw messages from Kafka

        Returns:
            Deduplicated list
        """
        seen_hashes = set()
        deduplicated = []

        for msg in messages:
            try:
                # Validate message structure
                prompt = msg.get("prompt", "")
                chosen = msg.get("chosen_response", "")
                rejected = msg.get("rejected_response", "")

                if not prompt or not chosen or not rejected:
                    log.warning("Skipping incomplete message", msg_keys=list(msg.keys()))
                    continue

                # Compute hash
                pair_str = f"{prompt}||{chosen}||{rejected}"
                pair_hash = hashlib.sha256(pair_str.encode()).hexdigest()

                if pair_hash in seen_hashes:
                    self.stats["duplicate_messages"] += 1
                else:
                    seen_hashes.add(pair_hash)
                    deduplicated.append(msg)
                    self.stats["valid_messages"] += 1

            except Exception as e:
                log.warning("Error processing message", error=str(e))

        log.info(
            "Deduplication complete",
            valid_messages=self.stats["valid_messages"],
            duplicates=self.stats["duplicate_messages"],
        )

        self.stats["dedup_rate"] = (
            self.stats["duplicate_messages"]
            / max(self.stats["total_raw_messages"], 1)
        )

        return deduplicated

    def _filter(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out low-quality examples.

        Criteria:
        - Min response length: 50 chars
        - Max response length: 4096 chars
        - No user bias: max 5% from single user

        Args:
            messages: Messages to filter

        Returns:
            Filtered list
        """
        MIN_LENGTH = 50
        MAX_LENGTH = 4096
        MAX_USER_PCT = 0.05

        # Filter by length
        filtered = []
        for msg in messages:
            chosen_len = len(msg.get("chosen_response", ""))
            rejected_len = len(msg.get("rejected_response", ""))

            if chosen_len < MIN_LENGTH or rejected_len < MIN_LENGTH:
                self.stats["filtered_messages"] += 1
                continue

            if chosen_len > MAX_LENGTH:
                msg["chosen_response"] = msg["chosen_response"][:MAX_LENGTH]
            if rejected_len > MAX_LENGTH:
                msg["rejected_response"] = msg["rejected_response"][:MAX_LENGTH]

            filtered.append(msg)

        # Check for user bias
        user_counts = defaultdict(int)
        for msg in filtered:
            user_counts[msg.get("user_id", "anonymous")] += 1

        max_user_count = max(user_counts.values()) if user_counts else 0
        user_bias_pct = max_user_count / max(len(filtered), 1)

        if user_bias_pct > MAX_USER_PCT:
            log.warning(
                "User bias detected",
                top_user_pct=f"{user_bias_pct:.1%}",
                threshold=f"{MAX_USER_PCT:.1%}",
            )

        self.stats["final_pairs"] = len(filtered)
        self.stats["filter_rate"] = self.stats["filtered_messages"] / max(
            self.stats["valid_messages"], 1
        )

        log.info(
            "Filtering complete",
            remaining=len(filtered),
            filtered_out=self.stats["filtered_messages"],
        )

        return filtered

    def _train_val_split(self, messages: List[Dict[str, Any]], train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
        """
        Split messages into train and validation sets.

        Args:
            messages: Filtered messages
            train_ratio: Fraction for training (default 0.9)

        Returns:
            Tuple of (train_pairs, val_pairs)
        """
        # Shuffle
        import random

        random.shuffle(messages)

        split_idx = int(len(messages) * train_ratio)
        train = messages[:split_idx]
        val = messages[split_idx:]

        self.stats["train_pairs"] = len(train)
        self.stats["val_pairs"] = len(val)

        log.info(
            "Train/val split complete",
            train=len(train),
            val=len(val),
            ratio=f"{train_ratio:.1%}/{1-train_ratio:.1%}",
        )

        return train, val

    def _save_to_s3(
        self, train_pairs: List[Dict], val_pairs: List[Dict]
    ) -> Tuple[str, str]:
        """
        Save train and val pairs to S3 as JSONL.

        Args:
            train_pairs: Training pairs
            val_pairs: Validation pairs

        Returns:
            Tuple of (train_s3_path, val_s3_path)
        """
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        prefix = f"preference-data/{date_str}"

        try:
            # Save train data
            train_data = {"examples": train_pairs}
            train_key = f"{prefix}/train.jsonl"
            train_path = self.s3_client.upload_json(train_key, train_data)

            # Save val data
            val_data = {"examples": val_pairs}
            val_key = f"{prefix}/val.jsonl"
            val_path = self.s3_client.upload_json(val_key, val_data)

            log.info(
                "Saved to S3",
                train_path=train_path,
                val_path=val_path,
                train_count=len(train_pairs),
                val_count=len(val_pairs),
            )

            return train_path, val_path

        except S3ClientError as e:
            log.error("Failed to save to S3", error=str(e))
            raise AggregationError(f"S3 save failed: {str(e)}") from e

    def _log_to_mlflow(self) -> None:
        """Log aggregation statistics to MLflow."""
        try:
            run_id = self.mlflow_client.start_run(
                experiment_name="nightly-pipeline",
                run_name="aggregation",
            )

            self.mlflow_client.log_params(
                {
                    "kafka_topic": self.settings.kafka_topic_feedback,
                    "train_ratio": 0.9,
                }
            )

            self.mlflow_client.log_metrics(
                {
                    "total_raw_messages": self.stats["total_raw_messages"],
                    "valid_messages": self.stats["valid_messages"],
                    "duplicate_messages": self.stats["duplicate_messages"],
                    "filtered_messages": self.stats["filtered_messages"],
                    "final_pairs": self.stats["final_pairs"],
                    "train_pairs": self.stats["train_pairs"],
                    "val_pairs": self.stats["val_pairs"],
                    "dedup_rate": self.stats["dedup_rate"],
                    "filter_rate": self.stats["filter_rate"],
                }
            )

            self.mlflow_client.end_run(status="FINISHED")
            log.info("Aggregation stats logged to MLflow", run_id=run_id)

        except MLflowClientError as e:
            log.warning("Failed to log to MLflow (non-blocking)", error=str(e))


def main():
    """Entrypoint for aggregator."""
    import sys

    try:
        from src.infra import setup_logging

        setup_logging(environment="production")

        aggregator = Aggregator()
        train_path, val_path = aggregator.aggregate_feedback()

        log.info("Aggregation successful", train=train_path, val=val_path)
        sys.exit(0)

    except AggregationError as e:
        log.error("Aggregation failed", error=str(e))
        sys.exit(1)
    except Exception as e:
        log.error("Unexpected error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
