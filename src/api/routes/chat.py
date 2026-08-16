"""Chat completion endpoint."""

import time
from typing import Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import structlog

from src.infra import get_logger
from src.infra.metrics import model_inference_duration
from src.models.config import get_settings

log = get_logger(__name__)
router = APIRouter()

# ============================================================================
# Pydantic Models
# ============================================================================


class ChatCompletionRequest(BaseModel):
    """OpenAI-format chat completion request."""

    prompt: str = Field(..., description="Input prompt to the model")
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False, description="Enable streaming response")


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list
    usage: dict


# ============================================================================
# Model Cache (stub for production, would load actual models)
# ============================================================================

_model_cache = {}


def get_champion_model():
    """
    Get or load the champion model.

    In production, this would load the LoRA adapter from S3 and merge with base model.
    For now, returns a stub.
    """
    if "champion" not in _model_cache:
        settings = get_settings()
        log.info(
            "Loading champion model (stub)",
            adapter_path=settings.champion_adapter_path,
        )
        _model_cache["champion"] = {
            "version": "v1.0",
            "adapter_path": settings.champion_adapter_path,
            "type": "stub",
        }
    return _model_cache["champion"]


# ============================================================================
# Chat Completion Endpoint
# ============================================================================


@router.post("/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-format chat completion endpoint.

    Accepts a prompt and streams or returns the response from the champion model.
    """
    import uuid
    from datetime import datetime

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        log.info(
            "Chat request received",
            request_id=request_id,
            prompt_length=len(request.prompt),
            max_tokens=request.max_tokens,
            stream=request.stream,
        )

        # Load model (cached)
        model = get_champion_model()
        model_version = model["version"]

        # Generate response (stub - just echo with prefix)
        response_text = f"[Champion {model_version}] Response to: {request.prompt[:50]}..."
        duration = time.time() - start_time

        log.info(
            "Chat response generated",
            request_id=request_id,
            response_length=len(response_text),
            duration_ms=round(duration * 1000, 2),
            model_version=model_version,
        )

        # Record inference latency
        model_inference_duration.labels(model_version=model_version).observe(duration)

        if request.stream:
            # Stream response via SSE
            async def stream_response():
                """Stream response in chunks."""
                chunk_size = 10
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i : i + chunk_size]
                    yield f"data: {{'delta': '{chunk}'}}\n\n"
                    await asyncio.sleep(0.01)  # Simulate streaming delay

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        else:
            # Return full response
            return ChatCompletionResponse(
                id=f"chatcmpl-{request_id}",
                created=int(datetime.utcnow().timestamp()),
                model=model_version,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": len(request.prompt.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(request.prompt.split()) + len(response_text.split()),
                },
            )

    except Exception as e:
        log.error(
            "Chat completion failed",
            request_id=request_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


import asyncio
