"""Prometheus metrics definitions."""

from prometheus_client import Counter, Histogram, Gauge

# ============================================================================
# HTTP Request Metrics
# ============================================================================
request_count = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
)

# ============================================================================
# Model Metrics
# ============================================================================
model_inference_duration = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration",
    ["model_version"],
)

current_champion_version = Gauge(
    "current_champion_version",
    "Current deployed champion model version",
    ["version"],
)

# ============================================================================
# Feedback Metrics
# ============================================================================
kafka_feedback_received = Counter(
    "kafka_feedback_received_total",
    "Total feedback events received",
    ["feedback_type"],
)
