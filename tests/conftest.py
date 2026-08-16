"""Pytest configuration and fixtures for integration tests."""

import json
import time
import pytest
import requests
from pathlib import Path
import tempfile
from typing import Dict, Any, Generator

import structlog
from confluent_kafka import Producer, Consumer
from confluent_kafka.admin import AdminClient, NewTopic

log = structlog.get_logger(__name__)


# =========================================================================
# SERVICE HEALTH CHECKS
# =========================================================================

def wait_for_service(url: str, timeout: int = 60, interval: int = 2) -> bool:
    """Wait for a service to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:
                log.info(f"Service ready: {url}")
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(interval)

    log.error(f"Service not ready after {timeout}s: {url}")
    return False


def kafka_ready(bootstrap_servers: str = "localhost:9092", timeout: int = 60) -> bool:
    """Check if Kafka is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            admin = AdminClient({"bootstrap.servers": bootstrap_servers})
            metadata = admin.list_topics(timeout=5)
            log.info("Kafka is ready")
            return True
        except Exception as e:
            log.debug(f"Kafka not ready: {e}")
            time.sleep(2)

    log.error("Kafka not ready after timeout")
    return False


# =========================================================================
# FIXTURES
# =========================================================================

def redis_ready(host: str = "localhost", port: int = 6380, timeout: int = 60) -> bool:
    """Check if Redis is ready."""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                log.info("Redis is ready")
                return True
        except Exception as e:
            log.debug(f"Redis not ready: {e}")
        time.sleep(2)

    log.error("Redis not ready after timeout")
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_services():
    """Setup and verify all services are running."""
    services = {
        "Kafka": ("localhost:9095", kafka_ready),
        "Redis": ("localhost:6380", redis_ready),
        "MLflow": ("http://localhost:5001", lambda: wait_for_service("http://localhost:5001")),
        "FastAPI": ("http://localhost:8001", lambda: wait_for_service("http://localhost:8001")),
    }

    log.info("Checking service availability...")
    for name, (endpoint, checker) in services.items():
        if not checker():
            pytest.skip(f"{name} not available at {endpoint}")
        log.info(f"✓ {name} is ready")

    yield

    log.info("Cleanup after test session")


@pytest.fixture
def kafka_producer() -> Generator[Producer, None, None]:
    """Kafka producer fixture."""
    producer = Producer({"bootstrap.servers": "localhost:9092"})
    yield producer
    producer.flush()


@pytest.fixture
def kafka_consumer(topic: str = "feedback.events") -> Generator[Consumer, None, None]:
    """Kafka consumer fixture."""
    consumer = Consumer({
        "bootstrap.servers": "localhost:9092",
        "group.id": f"test-consumer-{int(time.time())}",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([topic])
    yield consumer
    consumer.close()


@pytest.fixture
def kafka_admin_client() -> AdminClient:
    """Kafka admin client fixture."""
    return AdminClient({"bootstrap.servers": "localhost:9092"})


@pytest.fixture
def temp_s3_dir() -> Generator[Path, None, None]:
    """Temporary S3 mock directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_feedback_message() -> Dict[str, Any]:
    """Generate a mock feedback message."""
    return {
        "event_id": "test-event-1",
        "timestamp": int(time.time()),
        "user_id": "test_user_1",
        "prompt": "How can I improve my Python skills?",
        "chosen_response": (
            "Here's a comprehensive answer to 'How can I improve my Python skills?': "
            "This is a high-quality response that provides detailed information "
            "and practical advice. It covers multiple aspects of the topic."
        ),
        "rejected_response": (
            "Response to 'How can I improve my Python skills?': "
            "This is a lower quality response. It's shorter and less detailed."
        ),
        "feedback_type": "thumbs_up",
    }


@pytest.fixture
def mock_training_data() -> Dict[str, Any]:
    """Generate mock training data."""
    return {
        "examples": [
            {
                "prompt": f"Question {i}",
                "preferred": f"This is a high-quality response to question {i} with detailed information and examples.",
                "rejected": f"This is a low-quality response to question {i}.",
            }
            for i in range(10)
        ]
    }


@pytest.fixture
def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.app import app
    return TestClient(app)


@pytest.fixture
def mlflow_client():
    """MLflow client for tests."""
    from src.infra.mlflow_client import MLflowClient
    client = MLflowClient(tracking_uri="http://localhost:5000")
    return client


# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================

def produce_messages(producer: Producer, topic: str, messages: list[Dict[str, Any]]) -> None:
    """Produce messages to Kafka topic."""
    for msg in messages:
        producer.produce(topic, json.dumps(msg).encode("utf-8"))
    producer.flush()
    log.info(f"Produced {len(messages)} messages to {topic}")


def consume_messages(consumer: Consumer, timeout_ms: int = 5000, max_messages: int = 100) -> list[Dict[str, Any]]:
    """Consume messages from Kafka topic."""
    messages = []
    start = time.time()

    while len(messages) < max_messages:
        msg = consumer.poll(timeout_ms / 1000)

        if msg is None:
            if time.time() - start > timeout_ms / 1000 * 2:
                break
            continue

        if msg.error():
            log.error(f"Kafka error: {msg.error()}")
            continue

        try:
            messages.append(json.loads(msg.value().decode("utf-8")))
        except json.JSONDecodeError as e:
            log.warning(f"Failed to decode message: {e}")

    log.info(f"Consumed {len(messages)} messages")
    return messages
