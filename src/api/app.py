"""FastAPI application for DPO Continuous Learning Loop."""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import structlog

from src.models.config import get_settings
from src.infra import setup_logging, get_logger
from src.infra.metrics import (
    request_count,
    request_duration,
    model_inference_duration,
    current_champion_version,
    kafka_feedback_received,
)
from src.api.routes import chat, feedback, admin

# Setup logging
setup_logging(environment="production")
log = get_logger(__name__)

# ============================================================================
# Lifespan Context Manager (Startup/Shutdown)
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    settings = get_settings()
    log.info(
        "Application starting",
        environment=settings.environment,
        api_host=settings.api_host,
        api_port=settings.api_port,
    )

    # Load champion model version (stub for now)
    current_champion_version.labels(version="v1.0").set(1)

    yield

    # Shutdown
    log.info("Application shutting down")


# ============================================================================
# FastAPI App Creation
# ============================================================================

settings = get_settings()

app = FastAPI(
    title="DPO Continuous Learning Loop",
    description="Real-time preference learning with automated training and deployment",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Logging Middleware
# ============================================================================


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    start_time = time.time()

    # Log request
    log.debug(
        "Request received",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else "unknown",
    )

    try:
        response = await call_next(request)
    except Exception as e:
        log.error(
            "Request failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            exc_info=True,
        )
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500,
        ).inc()
        raise

    # Record metrics
    duration = time.time() - start_time
    request_duration.labels(method=request.method, endpoint=request.url.path).observe(duration)
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()

    # Log response
    log.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )

    return response


# ============================================================================
# Routes
# ============================================================================

# Root redirect to Swagger UI
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to Swagger UI."""
    return RedirectResponse(url="/docs")


# Health check endpoint
@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_version": "v1.0",
        "environment": settings.environment,
    }


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics() -> str:
    """Export Prometheus metrics."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    return generate_latest()


# Include route modules
app.include_router(chat.router, prefix="/v1", tags=["chat"])
app.include_router(feedback.router, tags=["feedback"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])


# ============================================================================
# Startup Banner
# ============================================================================


def print_startup_banner():
    """Print startup information (only output besides logs)."""
    print("\n" + "=" * 80)
    print("DPO Continuous Learning Loop - FastAPI Server")
    print("=" * 80)
    print(f"Environment: {settings.environment}")
    print(f"API Server: http://{settings.api_host}:{settings.api_port}")
    print(f"Swagger UI: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"Metrics: http://{settings.api_host}:{settings.api_port}/metrics")
    print(f"Health: http://{settings.api_host}:{settings.api_port}/health")
    print("=" * 80 + "\n")


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    log.error(
        "Unhandled exception",
        method=request.method,
        path=request.url.path,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    print_startup_banner()
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
