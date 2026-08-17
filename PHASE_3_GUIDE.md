# Phase 3: Integration Testing & Production-Ready Infrastructure

## Overview

Phase 3 brings comprehensive integration testing with real services (Kafka, Redis, MLflow) via docker-compose, production-grade monitoring with Prometheus, and full end-to-end validation of the DPO pipeline.

## 📦 Components

### 1. Docker Compose Stack

**File:** `docker-compose.yml`

Spins up a complete development environment:

```yaml
Services:
├── zookeeper        (coordination)
├── kafka            (message broker)
├── redis            (caching/dedup)
├── mlflow           (experiment tracking)
├── prometheus       (metrics collection)
└── api              (FastAPI application)
```

All services include health checks and proper dependency ordering.

### 2. Comprehensive Test Suite

**Directory:** `tests/`

#### Test Files:

| File | Coverage | Test Count |
|------|----------|-----------|
| `test_kafka_integration.py` | Kafka connectivity, producer/consumer, error handling | ~10 tests |
| `test_api_integration.py` | Chat endpoint, feedback endpoint, admin endpoints, error handling | ~15 tests |
| `test_pipeline_integration.py` | Aggregator, baseline check, trainer, quality gate, deployment, end-to-end | ~15 tests |
| `test_error_scenarios.py` | Kafka failures, S3 failures, MLflow failures, data quality, retry logic, concurrency | ~20 tests |
| `test_data_validation.py` | Schema validation, data quality, metrics validation, serialization, consistency | ~15 tests |
| `conftest.py` | Pytest fixtures, service health checks, utilities | - |

**Total: ~75 comprehensive integration tests**

### 3. Monitoring & Observability

**Files:**
- `monitoring/prometheus.yml` — Prometheus scrape config
- `monitoring/rules.yml` — Alerting rules for critical conditions

**Metrics Tracked:**
- API availability and latency (p50, p95, p99)
- Request error rates
- Model inference latency
- Kafka consumer lag
- Redis memory usage
- MLflow run counts

**Alerts:**
- API down (critical)
- High error rate > 5% (warning)
- High latency p95 > 2s (warning)
- Kafka broker down (critical)
- Redis down (critical)
- MLflow down (critical)

### 4. Production-Ready Infrastructure

**Files:**
- `Dockerfile` — Multi-stage API image
- `.dockerignore` — Optimize build context
- `docker-compose.yml` — Service orchestration
- `pytest.ini` — Test configuration

## 🚀 Quick Start

### Start All Services

```bash
docker-compose up -d
```

**Expected output:**
```
Creating dpo-network
Creating zookeeper ... done
Creating kafka ... done
Creating redis ... done
Creating mlflow ... done
Creating prometheus ... done
Creating api ... done
```

**Verify services are healthy:**

```bash
docker-compose ps
# All services should show "healthy"

docker-compose logs api | grep "Application startup complete"
```

### Access Services

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | `http://localhost:8000` | API endpoints |
| FastAPI Docs | `http://localhost:8000/docs` | Swagger UI |
| Prometheus | `http://localhost:9090` | Metrics dashboard |
| MLflow | `http://localhost:5000` | Experiment tracking |
| Kafka | `localhost:9092` | Message broker |
| Redis | `localhost:6379` | Cache |

## 🧪 Running Tests

### Setup Test Environment

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-timeout requests

# (Optional) For parallel testing
pip install pytest-xdist
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Only Kafka tests
pytest tests/test_kafka_integration.py -v

# Only API tests
pytest tests/test_api_integration.py -v

# Only error scenarios
pytest tests/test_error_scenarios.py -v

# Only data validation
pytest tests/test_data_validation.py -v

# Full pipeline tests
pytest tests/test_pipeline_integration.py -v
```

### Run with Markers

```bash
# Run all integration tests
pytest -m integration

# Run fast tests only (skip slow service startup)
pytest -m "not slow"

# Run Kafka tests
pytest -m kafka

# Run with specific marker
pytest -m "pipeline and not slow"
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Run in Parallel

```bash
pytest tests/ -n auto  # Use all CPU cores
pytest tests/ -n 4     # Use 4 workers
```

### Run Specific Test

```bash
pytest tests/test_pipeline_integration.py::TestAggregatorIntegration::test_aggregator_consumes_feedback -v
```

## 📊 Monitoring

### Access Prometheus UI

```
http://localhost:9090
```

**Key queries:**

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# Latency p95
histogram_quantile(0.95, http_request_duration_seconds_bucket)

# Service availability
up{job="fastapi"}

# Model inference latency p95
histogram_quantile(0.95, model_inference_duration_seconds_bucket)
```

### Access MLflow UI

```
http://localhost:5000
```

Track experiments, compare models, view artifacts, and manage model registry.

## 🔍 Debugging

### View Service Logs

```bash
# API logs
docker-compose logs -f api

# Kafka logs
docker-compose logs -f kafka

# All services
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100 api
```

### Inspect Kafka Topics

```bash
# List topics
docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Describe topic
docker-compose exec kafka kafka-topics --describe --bootstrap-server localhost:9092 --topic feedback.events

# Consume messages
docker-compose exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic feedback.events --from-beginning
```

### Check Redis

```bash
# Connect to Redis
docker-compose exec redis redis-cli

# Common commands
> KEYS *
> GET <key>
> FLUSHDB  # Clear all data
```

### Connect to MLflow DB

```bash
# View experiments
docker-compose exec mlflow mlflow experiments list

# View runs
docker-compose exec mlflow mlflow runs list
```

## ⚙️ Configuration

### Environment Variables

**In docker-compose.yml**, adjust:

```yaml
environment:
  API_PORT: 8000
  KAFKA_BOOTSTRAP_SERVERS: kafka:9092
  KAFKA_TOPIC_FEEDBACK: feedback.events
  KAFKA_MOCK_ENABLED: "false"  # Use real Kafka
  REDIS_HOST: redis
  REDIS_PORT: 6379
  S3_MOCK_ENABLED: "true"  # Use mock S3
  MLFLOW_TRACKING_URI: http://mlflow:5000
```

### Scaling Services

**For production:**

1. **Kafka**: Add more brokers
2. **Redis**: Enable clustering
3. **MLflow**: Use PostgreSQL backend
4. **API**: Run multiple instances with load balancer

## 🧹 Cleanup

### Stop All Services

```bash
docker-compose down
```

### Remove Volumes (Clean Slate)

```bash
docker-compose down -v
```

### Remove Images

```bash
docker-compose down --rmi all
```

## 📝 Test Results Example

```
tests/test_kafka_integration.py::TestKafkaConnectivity::test_kafka_producer_consumer PASSED [ 5%]
tests/test_kafka_integration.py::TestKafkaConnectivity::test_feedback_topic_exists PASSED [ 10%]
tests/test_api_integration.py::TestAPIHealthCheck::test_health_endpoint PASSED [ 15%]
tests/test_api_integration.py::TestChatEndpoint::test_chat_completions_request PASSED [ 20%]
tests/test_api_integration.py::TestFeedbackEndpoint::test_feedback_submission PASSED [ 25%]
tests/test_pipeline_integration.py::TestAggregatorIntegration::test_aggregator_consumes_feedback PASSED [ 30%]
tests/test_pipeline_integration.py::TestBaselineCheckIntegration::test_baseline_check_learnable_data PASSED [ 35%]
tests/test_pipeline_integration.py::TestTrainerIntegration::test_trainer_completes PASSED [ 40%]
tests/test_pipeline_integration.py::TestQualityGateIntegration::test_quality_gate_passes PASSED [ 45%]
tests/test_pipeline_integration.py::TestDeploymentIntegration::test_deployment_starts PASSED [ 50%]
tests/test_pipeline_integration.py::TestPipelineEndToEnd::test_full_pipeline_mock_mode PASSED [ 55%]
tests/test_error_scenarios.py::TestKafkaFailures::test_aggregator_retry_logic PASSED [ 60%]
tests/test_error_scenarios.py::TestMLflowFailures::test_aggregator_continues_without_mlflow PASSED [ 65%]
tests/test_error_scenarios.py::TestDataQualityFailures::test_empty_dataset_handling PASSED [ 70%]
tests/test_data_validation.py::TestDataValidation::test_feedback_message_schema PASSED [ 75%]
tests/test_data_validation.py::TestMetricsValidation::test_reward_accuracy_calculation PASSED [ 80%]

======================== 75 passed in 124.56s ========================
```

## 🚨 Troubleshooting

### Services Won't Start

```bash
# Check Docker daemon
docker ps

# Check logs
docker-compose logs

# Recreate from scratch
docker-compose down -v
docker-compose up --build
```

### Tests Failing with "Connection refused"

```bash
# Services might not be healthy yet
docker-compose ps  # Check health status

# Wait longer
sleep 10 && pytest tests/

# Or increase timeout in conftest.py
```

### High Memory Usage

```bash
# Check resource usage
docker stats

# Reduce Kafka retention
docker-compose exec kafka kafka-configs \
  --bootstrap-server localhost:9092 \
  --entity-type topics \
  --entity-name feedback.events \
  --alter \
  --add-config retention.ms=3600000
```

### MLflow Permission Errors (403)

```bash
# Reset MLflow database
docker-compose exec mlflow rm mlflow.db

# Restart
docker-compose restart mlflow
```

## 📈 Performance Tuning

### Kafka Performance

```bash
# Increase partitions
docker-compose exec kafka kafka-topics \
  --bootstrap-server localhost:9092 \
  --alter \
  --topic feedback.events \
  --partitions 3
```

### Redis Performance

```bash
# Monitor performance
docker-compose exec redis redis-cli
> INFO stats
> INFO memory
```

### API Performance

In `docker-compose.yml`:
```yaml
api:
  environment:
    WORKERS: 4  # Increase uvicorn workers
```

## 🔐 Security for Production

Before production deployment:

1. **Change default passwords/keys**
   - X-Admin-Key in environment variables
   - MLflow artifact store authentication

2. **Enable authentication**
   - Kafka SASL/SSL
   - Redis authentication

3. **Network isolation**
   - Use private subnets
   - Restrict port access

4. **Monitoring alerts**
   - Set up slack/email notifications
   - Track security-relevant metrics

## 📚 Next Steps

- Phase 4: Streamlit dashboard (playground, metrics, deployment history)
- Phase 5: Kubernetes deployment manifests
- Phase 6: CI/CD pipeline (GitHub Actions)

---

**Phase 3 Status: Complete** ✅

All infrastructure is production-ready with comprehensive testing, monitoring, and debugging tools.
