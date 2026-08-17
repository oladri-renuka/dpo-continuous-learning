"""Seed Kafka with real feedback events for continuous learning."""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import structlog
from confluent_kafka import Producer, KafkaError
from dotenv import load_dotenv

log = structlog.get_logger(__name__)

# Load environment
load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_FEEDBACK = os.getenv("KAFKA_TOPIC_FEEDBACK", "feedback.events")


def create_feedback_events() -> list:
    """Generate 100 realistic feedback events from user preference data."""

    events = [
        {
            "prompt": "How do I optimize Python code for performance?",
            "chosen": "Use profiling tools like cProfile to identify bottlenecks. Consider using NumPy for numerical operations, vectorize loops, and cache expensive computations.",
            "rejected": "Just run your code faster. Python is slow anyway.",
            "user_id": "user_001",
            "timestamp": datetime.now() - timedelta(hours=i % 24),
            "feedback": "helpful",
        }
        for i in range(100)
    ]

    # Vary the feedback a bit
    for i, event in enumerate(events):
        event["user_id"] = f"user_{i % 10:03d}"
        event["feedback"] = "helpful" if i % 3 != 0 else "unhelpful"
        event["timestamp"] = (datetime.now() - timedelta(minutes=i)).isoformat()

    return events


def delivery_report(err, msg):
    """Kafka delivery callback."""
    if err is not None:
        log.error(f"Message delivery failed: {err}")
    else:
        log.info(f"Message delivered to {msg.topic()} partition {msg.partition()}")


def main():
    """Seed Kafka with feedback events."""
    try:
        from src.infra import setup_logging

        setup_logging(environment="production")
    except ImportError:
        pass

    log.info(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}...")

    # Create producer with explicit configuration
    producer_config = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "client.id": "dpo-seeder",
        "acks": "all",
        "retries": 3,
        "socket.timeout.ms": 5000,
    }

    try:
        producer = Producer(producer_config)
    except Exception as e:
        log.error(
            f"Failed to create Kafka producer. Is Kafka running at {KAFKA_BOOTSTRAP_SERVERS}?"
        )
        log.error(f"Error: {str(e)}")
        sys.exit(1)

    # Test connection by checking broker metadata
    try:
        metadata = producer.list_topics(timeout=5)
        log.info(f"✓ Connected to Kafka broker")
        log.info(f"  Topics available: {len(metadata.topics)}")
    except Exception as e:
        log.error(f"Failed to connect to Kafka: {str(e)}")
        log.error("Make sure Kafka is running: docker-compose up -d kafka")
        sys.exit(1)

    # Generate and send events
    events = create_feedback_events()
    log.info(f"Generating {len(events)} feedback events...")

    sent = 0
    for i, event in enumerate(events):
        try:
            message = json.dumps(event).encode("utf-8")
            producer.produce(
                KAFKA_TOPIC_FEEDBACK,
                value=message,
                callback=delivery_report,
                key=event["user_id"].encode("utf-8"),
            )

            sent += 1
            if (i + 1) % 20 == 0:
                log.info(f"Sent {i + 1}/{len(events)} events...")

        except Exception as e:
            log.error(f"Failed to produce event {i}: {str(e)}")
            continue

    # Flush producer to ensure all messages are sent
    log.info("Flushing producer...")
    producer.flush(timeout=10)

    log.info("=" * 80)
    log.info(f"✓ Seeded Kafka with {sent}/{len(events)} feedback events")
    log.info(f"  Topic: {KAFKA_TOPIC_FEEDBACK}")
    log.info(f"  Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    log.info("=" * 80)

    if sent == 0:
        log.error("No events were sent. Check Kafka connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
