"""Aggregator daemon: Consume feedback from Redis, dedup via Redis, write to disk."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import redis
import structlog
from dotenv import load_dotenv

log = structlog.get_logger(__name__)

# Load environment
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "86400"))
REDIS_CHANNEL = "feedback_events"

DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "./data/raw"))
PIPELINE_FEEDBACK_THRESHOLD = int(os.getenv("PIPELINE_FEEDBACK_THRESHOLD", "500"))


class FeedbackAggregator:
    """Consume feedback events via Redis pub/sub, dedup, and trigger pipeline."""

    def __init__(self):
        self.data_raw_dir = DATA_RAW_DIR
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Redis for pub/sub and deduplication
        self.redis_client = self._init_redis()

        # Track current day's file and message count
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.current_file = self.data_raw_dir / f"feedback_{self.current_date}.jsonl"
        self.message_count = 0

        log.info(f"FeedbackAggregator initialized")
        log.info(f"  Redis: {REDIS_HOST}:{REDIS_PORT}")
        log.info(f"  Channel: {REDIS_CHANNEL}")
        log.info(f"  Data dir: {self.data_raw_dir}")
        log.info(f"  Threshold: {PIPELINE_FEEDBACK_THRESHOLD} messages")

    def _init_redis(self) -> redis.Redis:
        """Initialize real Redis client."""
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            # Test connection
            client.ping()
            log.info(f"✓ Redis connected")
            return client
        except Exception as e:
            log.error(f"Failed to connect to Redis: {str(e)}")
            log.error(f"Is Redis running at {REDIS_HOST}:{REDIS_PORT}?")
            raise

    def _is_duplicate(self, message: dict) -> bool:
        """Check if message is duplicate using Redis + SHA256."""
        # Create a hash of the feedback content
        content_str = json.dumps(
            {
                "prompt": message.get("prompt", ""),
                "chosen": message.get("chosen", ""),
                "rejected": message.get("rejected", ""),
                "user_id": message.get("user_id", ""),
            },
            sort_keys=True,
        )
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()

        # Check if hash exists in Redis
        redis_key = f"feedback:hash:{content_hash}"
        if self.redis_client.exists(redis_key):
            log.debug(f"Duplicate detected: {content_hash[:8]}...")
            return True

        # Store hash in Redis with TTL
        self.redis_client.setex(redis_key, REDIS_TTL_SECONDS, "1")
        return False

    def _write_feedback(self, message: dict) -> None:
        """Write feedback to daily JSONL file."""
        # Ensure timestamp exists
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        # Rotate file if date changed
        new_date = datetime.now().strftime("%Y-%m-%d")
        if new_date != self.current_date:
            log.info(f"Date changed to {new_date}, rotating file")
            self.current_date = new_date
            self.current_file = self.data_raw_dir / f"feedback_{self.current_date}.jsonl"
            self.message_count = 0

        # Append to file
        with open(self.current_file, "a") as f:
            f.write(json.dumps(message) + "\n")

        self.message_count += 1
        log.info(
            f"Wrote feedback #{self.message_count} to {self.current_file.name}"
        )

    def _trigger_pipeline(self) -> None:
        """Trigger orchestrator when threshold is reached."""
        log.info("=" * 80)
        log.info(f"THRESHOLD REACHED: {self.message_count} messages accumulated")
        log.info(f"Triggering pipeline: scripts/run_pipeline.py")
        log.info("=" * 80)

        try:
            # Call orchestrator in subprocess with proper Python path
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path.cwd())

            result = subprocess.run(
                ["python", "scripts/run_pipeline.py"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env=env,
            )
            log.info(f"Pipeline completed successfully")
            log.info(f"Output: {result.stdout}")

            # Reset counter after successful pipeline run
            self.message_count = 0

        except subprocess.TimeoutExpired:
            log.error(f"Pipeline timed out (>1 hour)")
        except subprocess.CalledProcessError as e:
            log.error(f"Pipeline failed with exit code {e.returncode}")
            log.error(f"Stdout: {e.stdout}")
            log.error(f"Stderr: {e.stderr}")
        except Exception as e:
            log.error(f"Failed to trigger pipeline: {str(e)}")

    def run(self) -> None:
        """Run daemon: consume Redis, dedup, write, and trigger pipeline."""
        log.info("=" * 80)
        log.info("AGGREGATOR DAEMON STARTED")
        log.info("=" * 80)
        log.info(f"Listening on channel: {REDIS_CHANNEL}")

        # Subscribe to Redis channel
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(REDIS_CHANNEL)

        try:
            for message in pubsub.listen():
                # Skip subscription confirmation messages
                if message["type"] != "message":
                    continue

                try:
                    feedback = json.loads(message["data"])
                    log.debug(f"Received feedback: user={feedback.get('user_id')}")

                    # Check for duplicates
                    if self._is_duplicate(feedback):
                        log.debug(f"Skipped duplicate")
                        continue

                    # Write to disk
                    self._write_feedback(feedback)

                    # Check threshold
                    if self.message_count >= PIPELINE_FEEDBACK_THRESHOLD:
                        self._trigger_pipeline()

                except json.JSONDecodeError as e:
                    log.error(f"Failed to parse message: {str(e)}")
                    continue
                except Exception as e:
                    log.error(f"Error processing message: {str(e)}", exc_info=True)
                    continue

        except KeyboardInterrupt:
            log.info("Shutting down (Ctrl+C)...")
        finally:
            pubsub.close()
            self.redis_client.close()
            log.info("Aggregator daemon stopped")


def main():
    """Entry point."""
    try:
        from src.infra import setup_logging

        setup_logging(environment="production")
    except ImportError:
        pass

    aggregator = FeedbackAggregator()
    aggregator.run()


if __name__ == "__main__":
    main()
