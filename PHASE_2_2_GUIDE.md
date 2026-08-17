# Phase 2.2: FastAPI Server & Deployment - Quick Start Guide

## Files Generated

```
src/
├── api/
│   ├── __init__.py
│   ├── app.py                    # FastAPI server (Prometheus metrics, CORS, middleware)
│   ├── deployment.py             # Blue/Green deployer with canary rollout
│   └── routes/
│       ├── __init__.py
│       ├── chat.py               # POST /v1/chat/completions
│       ├── feedback.py           # POST /feedback (Kafka integration)
│       └── admin.py              # POST /admin/rollback, GET /admin/model-info
├── models/
│   └── config.py                 # Pydantic Settings (loads from .env)
.env.example                        # Environment configuration template
```

## Startup Instructions

### 1. Copy Environment Template

```bash
cp .env.example .env
# Edit .env if needed (defaults are fine for local testing)
```

### 2. Start the FastAPI Server

```bash
make run-api
```

Expected output:
```
================================================================================
DPO Continuous Learning Loop - FastAPI Server
================================================================================
Environment: development
API Server: http://0.0.0.0:8000
Swagger UI: http://0.0.0.0:8000/docs
Metrics: http://0.0.0.0:8000/metrics
Health: http://0.0.0.0:8000/health
================================================================================
```

The server is now running on `http://localhost:8000`

---

## Testing Endpoints

### Test 1: Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_version": "v1.0",
  "environment": "development"
}
```

### Test 2: Swagger UI

Open in browser:
```
http://localhost:8000/docs
```

This shows interactive API documentation.

### Test 3: Chat Completion (Non-Streaming)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "max_tokens": 256,
    "stream": false
  }'
```

Expected response:
```json
{
  "id": "chatcmpl-...",
  "object": "text_completion",
  "created": 1692864000,
  "model": "v1.0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "[Champion v1.0] Response to: What is machine learning?..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 12,
    "total_tokens": 17
  }
}
```

### Test 4: Submit Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain supervised learning",
    "chosen_response": "Supervised learning uses labeled data...",
    "rejected_response": "Supervised learning is about classification.",
    "feedback_type": "thumbs_up",
    "user_id": "user_123"
  }'
```

Expected response (202 Accepted, fire-and-forget):
```json
{
  "status": "accepted",
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-08-16T18:00:00.000000"
}
```

### Test 5: Metrics (Prometheus Format)

```bash
curl http://localhost:8000/metrics
```

Expected output (Prometheus format):
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{endpoint="/health",method="GET",status="200"} 1.0
http_requests_total{endpoint="/v1/chat/completions",method="POST",status="200"} 1.0
...
```

### Test 6: Admin - Get Model Info

```bash
curl http://localhost:8000/admin/model-info
```

Expected response:
```json
{
  "champion": {
    "version": "v1.0",
    "adapter_path": "s3://ml-artifacts/models/champion/adapter_model.bin",
    "deployed_at": "2026-08-16T18:00:00.000000",
    "metrics": {
      "reward_accuracy": 0.82,
      "dpo_win_rate": 0.73
    }
  },
  "deployment_history": []
}
```

### Test 7: Admin - Trigger Rollback (Requires Admin Key)

```bash
curl -X POST http://localhost:8000/admin/rollback \
  -H "X-Admin-Key: your-super-secret-admin-key-change-this"
```

Expected response:
```json
{
  "status": "rollback_triggered",
  "message": "Rolled back from v1.0 to v0.9",
  "timestamp": "2026-08-16T18:00:00.000000"
}
```

---

## Monitoring Logs

In another terminal, tail the structured logs:

```bash
# All logs are JSON-formatted and streamed to stdout
# You can pipe to jq for pretty-printing:
tail -f /tmp/app.log | jq '.'
```

Or filter by level:

```bash
grep '"level":"error"' /tmp/app.log  # Errors only
grep '"level":"warning"' /tmp/app.log # Warnings
```

---

## Architecture Summary

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Redirect to Swagger UI |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/v1/chat/completions` | POST | Chat completion (OpenAI format) |
| `/feedback` | POST | Submit user feedback (fire-and-forget to Kafka) |
| `/admin/rollback` | POST | Trigger rollback (admin only) |
| `/admin/model-info` | GET | Get champion model info |

### Key Features

✅ **Structured Logging**: All requests/responses logged as JSON  
✅ **Prometheus Metrics**: Request counts, latencies, model version tracking  
✅ **Error Handling**: Graceful degradation if Kafka/MLflow/S3 are down  
✅ **Fire-and-Forget Feedback**: Background tasks for non-blocking Kafka publishing  
✅ **CORS**: Configurable for development/production  
✅ **Admin Authentication**: X-Admin-Key header validation  
✅ **Mock Services**: All external services can run in mock mode  

---

## Environment Variables

All configuration comes from `.env` (loaded via Pydantic Settings):

```
API_HOST=0.0.0.0                                    # API listen address
API_PORT=8000                                       # API listen port
ADMIN_API_KEY=your-key                              # Admin authentication
KAFKA_BOOTSTRAP_SERVERS=localhost:9092              # Kafka brokers
KAFKA_TOPIC_FEEDBACK=feedback.events                # Kafka topic
KAFKA_MOCK_ENABLED=true                             # Mock Kafka (no real broker needed)
S3_MOCK_ENABLED=true                                # Mock S3 (local file storage)
MLFLOW_TRACKING_URI=http://localhost:5000           # MLflow server
ENVIRONMENT=development                             # dev/staging/production
```

---

## Next Steps

Once you've tested the endpoints:

1. **Phase 2.3**: Generate the aggregator, trainer, and Kafka consumer
2. **Phase 2.4**: Generate the orchestrator script that chains everything
3. **Phase 3**: Integration testing with docker-compose (Kafka, Redis, MLflow)
4. **Phase 4**: Streamlit dashboard
