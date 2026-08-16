"""Feedback submission endpoint."""

import json
import uuid
import asyncio
from typing import Optional
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from fastapi import APIRouter, BackgroundTasks, HTTPException
import structlog

from src.infra import get_logger
from src.infra.metrics import kafka_feedback_received
from src.models.config import get_settings

log = get_logger(__name__)
router = APIRouter()

# ============================================================================
# Pydantic Models
# ============================================================================


class FeedbackRequest(BaseModel):
    """User feedback submission."""

    prompt: str = Field(..., description="The prompt that was shown to user")
    chosen_response: str = Field(..., description="Response user preferred")
    rejected_response: str = Field(..., description="Response user rejected")
    feedback_type: str = Field(..., description="thumbs_up or thumbs_down")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")


class FeedbackResponse(BaseModel):
    """Feedback submission response."""

    status: str
    event_id: str
    timestamp: str


# ============================================================================
# Kafka Client (simplified, non-blocking)
# ============================================================================

_kafka_queue = []  # Local retry queue if Kafka is down


async def publish_to_kafka(event: dict):
    """
    Publish feedback event to Kafka (fire-and-forget, production only).

    If Kafka is down, raises exception.
    """
    settings = get_settings()

    try:
        from confluent_kafka import Producer

        producer = Producer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "client.id": "dpo-api",
            }
        )

        producer.produce(
            settings.kafka_topic_feedback,
            json.dumps(event).encode("utf-8"),
            key=event["event_id"].encode("utf-8"),
        )
        producer.flush(timeout=5)

        log.info(
            "Event published to Kafka",
            topic=settings.kafka_topic_feedback,
            event_id=event.get("event_id"),
        )
    except Exception as e:
        log.error(
            "Failed to publish to Kafka",
            error=str(e),
            event_id=event.get("event_id"),
        )
        _kafka_queue.append(event)
        raise


def retry_queued_events():
    """Attempt to retry queued events (background task)."""
    if not _kafka_queue:
        return

    log.info("Retrying queued Kafka events", queue_size=len(_kafka_queue))
    # In production, would implement proper retry logic with exponential backoff


# ============================================================================
# Feedback Endpoint
# ============================================================================


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit user feedback.

    Accepts preference feedback and publishes to Kafka for aggregation.
    Returns immediately with 202 Accepted (fire-and-forget).
    """
    event_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    try:
        # Validate request
        if request.feedback_type not in ("thumbs_up", "thumbs_down"):
            raise ValueError(f"Invalid feedback_type: {request.feedback_type}")

        # Create event
        event = {
            "event_id": event_id,
            "timestamp": timestamp,
            "user_id": request.user_id or "anonymous",
            "prompt": request.prompt,
            "chosen_response": request.chosen_response,
            "rejected_response": request.rejected_response,
            "feedback_type": request.feedback_type,
        }

        log.info(
            "Feedback received",
            event_id=event_id,
            feedback_type=request.feedback_type,
            prompt_length=len(request.prompt),
        )

        # Publish to Kafka (fire-and-forget via background task)
        background_tasks.add_task(publish_to_kafka, event)

        # Record metric
        kafka_feedback_received.labels(feedback_type=request.feedback_type).inc()

        # Return 202 Accepted (fire-and-forget)
        return FeedbackResponse(
            status="accepted",
            event_id=event_id,
            timestamp=timestamp,
        )

    except ValueError as e:
        log.error("Invalid feedback request", error=str(e), event_id=event_id)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        log.error(
            "Failed to process feedback",
            error=str(e),
            event_id=event_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to process feedback")
