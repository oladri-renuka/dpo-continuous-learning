# Continuous Preference Learning Loop

A production-grade system for continuous alignment of LLMs through real-time user feedback, automated training, and zero-downtime deployment.

**Target Metrics**: Reward Model Accuracy > 72%, DPO Win-Rate > 55%

---

## Overview

This system implements a fully automated feedback loop:

1. **Collect** user preferences (thumbs up/down) in real-time via Kafka
2. **Aggregate** feedback nightly into preference pairs
3. **Train** a reward model + DPO-aligned LLM using QLoRA on RunPod GPUs
4. **Validate** against strict quality gates (no bad models deployed)
5. **Deploy** via Blue/Green strategy with automatic rollback
6. **Monitor** with Prometheus metrics and MLflow experiment tracking

All with **zero downtime**, **automatic recovery**, and **full reproducibility**.

---

## Key Features

✅ **Hard Quality Gates**: Metric thresholds are non-negotiable. Failed training = pipeline halt (no bypass flag).

✅ **Data Quality First**: Pre-training baseline check ensures data is learnable before expensive GPU training.

✅ **Error Resilience**: Exponential backoff retries, dead-letter queues, circuit breakers, and graceful degradation for every external service.

✅ **Local Development**: Full pipeline testable offline with mock Kafka, mock RunPod GPU, and mock MLflow.

✅ **Immutable Infrastructure**: Docker images contain all code; no runtime volume mounts required.

✅ **Structured Observability**: JSON logging, Prometheus metrics, MLflow lineage, and automatic RCA diagnostics.

✅ **Dashboard**: Streamlit UI with playground (side-by-side model comparison), metrics history, and deployment logs.

---

## Architecture at a Glance

```
User Feedback (Kafka)
    ↓
Real-time Ingestion (Kafka Consumer)
    ↓
Nightly Aggregation (Batch Job)
    ↓
Baseline Check (Data Quality Gate) → HALT if data is noisy
    ↓
Training on RunPod (QLoRA + DPO)
    ↓
Quality Gate (Validation) → HALT if metrics fail
    ↓
Blue/Green Deployment (Zero-downtime swap)
    ↓
Monitoring & Observability (Prometheus + MLflow + Structlog)
```

For detailed system design, see [docs/DESIGN.md](docs/DESIGN.md).

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Message Broker** | Apache Kafka | High-throughput, replay-capable, industry standard |
| **Cache Layer** | Redis | Sub-ms dedup, native TTL support |
| **Training** | Hugging Face (transformers, peft, trl) | Standard LLM fine-tuning ecosystem |
| **GPU Jobs** | RunPod Serverless | Cost-efficient, no infrastructure overhead |
| **Experiment Tracking** | MLflow | Free, offline support, model registry |
| **API Server** | FastAPI | High performance, async support |
| **Logging** | structlog + JSON | Machine-parseable logs for CloudWatch/Datadog |
| **Metrics** | Prometheus | Standard ops visibility |
| **Deployment** | Kubernetes + Custom Blue/Green Logic | Zero-downtime model swaps |
| **Dashboard** | Streamlit | Simple, no backend required |

---

## Project Structure

```
dpo-continuous-learning/
├── docs/
│   ├── DESIGN.md                    # Architecture, data flow, failure modes
│   └── API_REFERENCE.md
├── src/
│   ├── core/
│   │   ├── aggregator.py           # Nightly batch aggregation
│   │   ├── trainer.py              # QLoRA + DPO (RunPod entrypoint)
│   │   └── quality_gate.py          # Hard validation gates
│   ├── infra/
│   │   ├── kafka_client.py         # Consumer with DLQ
│   │   ├── redis_client.py         # Dedup + caching
│   │   ├── runpod_client.py        # GPU job submission
│   │   └── logging_config.py       # structlog setup
│   ├── api/
│   │   ├── app.py                  # FastAPI server
│   │   ├── routes/                 # Chat, feedback, admin endpoints
│   │   └── deployment.py           # Blue/Green logic
│   ├── ui/
│   │   └── app.py                  # Streamlit dashboard
│   └── models/
│       ├── schemas.py              # Pydantic validation
│       └── config.py               # Settings
├── tests/
│   ├── test_aggregator.py
│   ├── test_quality_gate.py
│   └── test_api.py
├── scripts/
│   ├── baseline_check.py           # Pre-training data quality gate
│   ├── diagnose.py                 # RCA tool for metric failures
│   ├── seed_data.py                # Mock data generator
│   └── run_consumer.py             # Kafka consumer entrypoint
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.trainer
│   └── docker-compose.dev.yml
├── config/
│   └── config.yaml                 # Dev/prod settings
├── Makefile
├── requirements.txt
├── .env.example
└── README.md                        # You are here
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git
- (Optional) RunPod API key for real GPU training

### Installation

```bash
# Clone repo
git clone <repo>
cd dpo-continuous-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
make install

# Copy environment template
cp .env.example .env
# Edit .env with your API keys (or leave blank for mock mode)
```

### Quick Start (Local Development with Mocks)

```bash
# Start Kafka, Redis, MLflow locally
docker-compose -f docker/docker-compose.dev.yml up -d

# Generate mock feedback data
python scripts/seed_data.py --num-samples 1000 --output /tmp/feedback.jsonl

# Run the Kafka consumer (with mock RunPod training)
RUNPOD_MOCK=true python -m src.core.aggregator --data-source /tmp/feedback.jsonl

# Start the FastAPI server
make run-api
# Opens at http://localhost:8000

# Start the Streamlit dashboard
streamlit run src/ui/app.py
# Opens at http://localhost:8501
```

### Running Tests

```bash
# Full test suite with coverage
make test

# Linting + formatting
make lint

# Run specific test file
pytest tests/test_quality_gate.py -v
```

---

## API Endpoints

### Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "champion",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "stream": false
  }'
```

### Feedback Submission

```bash
curl -X POST http://localhost:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "prompt": "Explain quantum computing",
    "responses": {
      "sft_response": "...",
      "rl_response": "..."
    },
    "preferred": "rl_response",
    "feedback_type": "thumbs_up"
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
# Response: {"status": "healthy", "model_version": "v1.2"}
```

### Metrics (Prometheus)

```bash
curl http://localhost:8000/metrics
# Prometheus-format metrics for Grafana scraping
```

For full API reference, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

---

## Running in Production

### Build Docker Images

```bash
# API server
docker build -f docker/Dockerfile.api -t dpo-api:latest .

# Training job (for RunPod)
docker build -f docker/Dockerfile.trainer -t dpo-trainer:latest .
```

### Deploy with Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Scale workers
kubectl scale deployment dpo-api --replicas=3
```

### Trigger Training Job

```bash
# Manually trigger (normally runs on cron, 02:00 UTC daily)
curl -X POST http://localhost:8000/admin/train-now
```

### Monitor Metrics

```bash
# Prometheus at http://your-prometheus:9090
# Grafana dashboard: Search for "DPO Continuous Learning"

# MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# Open http://localhost:5000
```

---

## Quality Gates Explained

### 1. Baseline Check (Pre-Training)

Runs `scripts/baseline_check.py` on aggregated feedback.

- Loads 1,000 samples of preference pairs
- Trains logistic regression on Sentence-BERT embeddings
- Validates accuracy > 55%
- If FAIL: Exits with code 1, halts pipeline. Data is intrinsically noisy.

**Result**: Ensures training data is learnable before expensive GPU costs.

### 2. Metric Validation (Post-Training)

Runs `src/core/quality_gate.py` after training completes.

- Loads newly trained LoRA adapter (Challenger)
- Loads current model in production (Champion)
- Evaluates on 100 held-out golden examples
- Computes:
  - **Reward Model Accuracy**: % of examples where predicted preferred > predicted rejected
  - **DPO Win-Rate**: % of examples where Challenger outperforms Champion
- Checks: Accuracy > 72% AND Win-Rate > 55%
- If FAIL: Raises `ModelDegradationError`, exits code 1, **no deployment happens**

**Result**: Guarantees only better models are deployed. Bad models are physically impossible to deploy.

---

## Failure Modes & Recovery

| Scenario | Response |
|----------|----------|
| **Kafka down** | Consumer retries with exponential backoff; uses local queue if available |
| **RunPod OOM** | Reduce batch size, LoRA rank; resume from checkpoint; fallback to smaller model |
| **Quality gate fails** | Log error, exit code 1, halt deployment; trigger RCA diagnostics |
| **Model degradation post-deployment** | Canary detects drop in success rate; auto-rollback to previous version |
| **Redis down** | Fall back to SQLite; sync with Redis when it's back online |
| **MLflow unavailable** | Log metrics locally; manually sync when MLflow is back |

See [docs/DESIGN.md § 4](docs/DESIGN.md) for detailed recovery strategies.

---

## Monitoring & Alerts

### Key Metrics

- **Reward Model Accuracy**: Should remain > 72%
- **DPO Win-Rate**: Should remain > 55%
- **API Success Rate**: Should remain > 99%
- **P99 Latency**: Should remain < 200ms
- **Kafka Consumer Lag**: Should remain < 10,000 messages

### Alert Rules

- **CRITICAL**: Quality gate failure → PagerDuty + Slack
- **CRITICAL**: Canary degradation during deployment → Auto-rollback + Slack
- **WARNING**: Kafka consumer lag > 10,000 → Slack #ops
- **WARNING**: Baseline check fails → Halt pipeline, manual review required

---

## Dashboard Features

### Playground Tab

- Type a prompt
- See real-time responses from Champion (prod) and Challenger (candidate)
- Compare outputs side-by-side
- Show which model generated which response

### Metrics Dashboard Tab

- Historical charts: Reward Model Accuracy, DPO Win-Rate (7-day rolling window)
- Fetch data from MLflow API
- Current health status: Pass/Fail indicator
- Trend analysis: Is accuracy improving over time?

### Deployment History Tab

- Table: Version | Timestamp | Metrics (Acc/Win-Rate) | Gate Status (Pass/Fail) | Deployment Status (Live/Rolled Back)
- Quick drill-down into any deployment
- Side-by-side diff: Old model outputs vs. new model outputs

---

## Troubleshooting

### "Data is intrinsically noisy" (baseline check fails)

**Symptom**: `scripts/baseline_check.py` reports accuracy < 55%

**Causes**:
- Feedback is mislabeled (user clicked wrong button)
- Responses are too similar (prompt doesn't create preference signal)
- Low-quality feedback data (spam, bots)

**Fix**:
1. Review 50 random feedback samples manually
2. Check for labeling errors
3. Adjust feedback UI to reduce ambiguity
4. Re-aggregate and re-run baseline check

### "Quality gate failed: Accuracy 68% < 72%"

**Symptom**: Training completes, but metric validation fails

**Causes**:
- Model is overfitting (train loss low, val loss high)
- Data distribution changed (user feedback shifted)
- Hyperparameters suboptimal

**Fix**:
1. Run `scripts/diagnose.py` to inspect train/val loss ratio
2. Increase LoRA dropout from 0.1 to 0.2
3. Reduce training epochs from 3 to 2
4. Add weight decay (0.01)
5. Retrain

### "Canary detected degradation, rolled back"

**Symptom**: Deployment started, but Challenger model showed 94% success rate (< 95% threshold)

**Causes**:
- Challenger model is less robust than Champion
- Metric validation gate was too permissive (should have caught this)

**Fix**:
1. Inspect Challenger outputs: Is it producing unsafe content?
2. Check for mode collapse: Does it repeat Champion's responses?
3. Increase golden eval set from 100 to 500 examples (stricter validation)
4. Retrain with tighter hyperparameters

---

## Contributing

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for contribution guidelines.

---

## License

[Your License Here]

---

## Contact

For issues, questions, or feedback:
- **Email**: ml-team@company.com
- **Slack**: #ml-platform
- **GitHub Issues**: [repo/issues](repo/issues)

---

## Appendix: Common Commands

```bash
# Development
make install              # Install dependencies
make lint                 # Ruff + Black
make test                 # Pytest with coverage
make run-api              # Start FastAPI server (http://localhost:8000)
make run-consumer         # Start Kafka consumer

# Docker
docker-compose -f docker/docker-compose.dev.yml up      # Local dev stack
docker build -f docker/Dockerfile.api -t dpo-api .      # Build API image
docker build -f docker/Dockerfile.trainer -t dpo-trainer . # Build trainer image

# Data & Testing
python scripts/seed_data.py --num-samples 1000           # Generate mock feedback
python scripts/baseline_check.py --data /tmp/feedback.jsonl # Pre-training validation
python scripts/diagnose.py --run-id abc123              # RCA for failed training

# Monitoring
mlflow ui --host 0.0.0.0 --port 5000                    # MLflow UI
streamlit run src/ui/app.py                              # Dashboard
curl http://localhost:8000/metrics                       # Prometheus metrics

# Admin
curl -X POST http://localhost:8000/admin/train-now       # Trigger training
curl -X POST http://localhost:8000/admin/rollback        # Manual rollback
```

---

## Roadmap

- [ ] Phase 2: Code generation (all src/ files)
- [ ] Phase 3: Local integration testing
- [ ] Phase 4: Dashboard UI
- [ ] Phase 5: Kubernetes manifests
- [ ] Phase 6: Production deployment playbook
- [ ] Multi-model support (switch base models)
- [ ] Active learning (prioritize uncertain feedback)
- [ ] Budget constraints (auto-pause training if cost exceeds threshold)

---

**Last Updated**: 2026-08-16
