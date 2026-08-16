"""Kafka client with retry logic, DLQ, and error handling."""

import json
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

log = structlog.get_logger(__name__)


class KafkaClientError(Exception):
    """Custom exception for Kafka operations."""
    pass


class KafkaConsumer:
    """Kafka consumer with automatic retries, dead-letter queue, and error handling."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "feedback.events",
        group_id: str = "dpo-feedback-consumer",
    ):
        """
        Initialize Kafka consumer (production only).

        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic to consume from
            group_id: Consumer group ID

        Raises:
            KafkaClientError: If Kafka broker is unreachable
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        self.dlq_topic = f"{topic}.dlq"

        self._init_consumer()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _init_consumer(self):
        """Initialize Kafka consumer with retries. Raises if broker unreachable."""
        try:
            from confluent_kafka import Consumer

            self.consumer = Consumer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "group.id": self.group_id,
                    "auto.offset.reset": "earliest",
                    "enable.auto.commit": True,
                    "max.poll.interval.ms": 300000,
                }
            )
            self.consumer.subscribe([self.topic])
            log.info(
                "Kafka consumer initialized",
                bootstrap_servers=self.bootstrap_servers,
                topic=self.topic,
                group_id=self.group_id,
            )
        except Exception as e:
            log.error("Failed to initialize Kafka consumer", error=str(e))
            raise KafkaClientError(f"Kafka broker unreachable at {self.bootstrap_servers}: {str(e)}") from e

    def consume_batch(self, num_messages: int = 1000, timeout_ms: int = 30000) -> List[Dict[str, Any]]:
        """
        Consume a batch of messages from Kafka.

        Args:
            num_messages: Maximum number of messages to consume
            timeout_ms: Timeout in milliseconds

        Returns:
            List of consumed messages (deserialized JSON)

        Raises:
            KafkaClientError: If consumption fails
        """
        try:
            log.info(
                "Starting Kafka batch consumption",
                topic=self.topic,
                num_messages=num_messages,
            )

            messages = []
            timeout_sec = timeout_ms / 1000

            from confluent_kafka import KafkaError

            while len(messages) < num_messages:
                msg = self.consumer.poll(timeout=min(1, timeout_sec))

                if msg is None:
                    break

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        log.debug("Reached end of partition")
                        break
                    else:
                        log.error("Kafka error", error=msg.error())
                        raise KafkaClientError(f"Kafka error: {msg.error()}")

                try:
                    payload = json.loads(msg.value().decode("utf-8"))
                    messages.append(payload)
                except json.JSONDecodeError as e:
                    log.warning(
                        "Failed to parse message, sending to DLQ",
                        error=str(e),
                        message=msg.value()[:100],
                    )
                    self._send_to_dlq(msg.value(), "json_decode_error", str(e))

            log.info(
                "Batch consumption complete",
                topic=self.topic,
                messages_consumed=len(messages),
            )
            return messages

        except KafkaClientError:
            raise
        except Exception as e:
            log.error("Consumption failed", error=str(e), exc_info=True)
            raise KafkaClientError(f"Failed to consume from Kafka: {str(e)}") from e

    def _send_to_dlq(self, message: bytes, error_type: str, error_detail: str) -> None:
        """Send failed message to dead-letter queue."""
        try:
            from confluent_kafka import Producer

            producer = Producer({"bootstrap.servers": self.bootstrap_servers})

            dlq_message = {
                "original_message": message.decode("utf-8", errors="replace"),
                "error_type": error_type,
                "error_detail": error_detail,
            }

            producer.produce(
                self.dlq_topic,
                json.dumps(dlq_message).encode("utf-8"),
            )
            producer.flush()

            log.warning(
                "Message sent to DLQ",
                dlq_topic=self.dlq_topic,
                error_type=error_type,
            )
        except Exception as e:
            log.error("Failed to send to DLQ", error=str(e))

    def close(self):
        """Close the consumer connection."""
        if self.consumer:
            self.consumer.close()
            log.info("Kafka consumer closed")


class KafkaProducer:
    """Kafka producer for publishing events."""

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        """
        Initialize Kafka producer (production only).

        Args:
            bootstrap_servers: Kafka broker addresses

        Raises:
            KafkaClientError: If Kafka broker is unreachable
        """
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self._init_producer()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _init_producer(self):
        """Initialize Kafka producer with retries. Raises if broker unreachable."""
        try:
            from confluent_kafka import Producer

            self.producer = Producer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "acks": "all",
                }
            )
            log.info("Kafka producer initialized", bootstrap_servers=self.bootstrap_servers)
        except Exception as e:
            log.error("Failed to initialize Kafka producer", error=str(e))
            raise KafkaClientError(f"Kafka broker unreachable at {self.bootstrap_servers}: {str(e)}") from e

    def produce(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Produce a message to Kafka.

        Args:
            topic: Topic to produce to
            message: Message payload (will be JSON serialized)
            key: Optional message key

        Returns:
            True if successful, False otherwise

        Raises:
            KafkaClientError: If production fails
        """
        try:
            self.producer.produce(
                topic,
                json.dumps(message).encode("utf-8"),
                key=key.encode("utf-8") if key else None,
            )
            self.producer.flush()
            log.debug("Message produced", topic=topic, key=key)
            return True
        except Exception as e:
            log.error("Failed to produce message", error=str(e), topic=topic)
            raise KafkaClientError(f"Failed to produce to {topic}: {str(e)}") from e

    def close(self):
        """Close the producer."""
        if self.producer:
            self.producer.flush()
            log.info("Kafka producer closed")
