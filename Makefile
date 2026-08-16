.PHONY: install lint test run-api run-consumer clean help docker-build docker-push docs

# Colors for output
BLUE=\033[0;34m
GREEN=\033[0;32m
RED=\033[0;31m
NC=\033[0m # No Color

help:
	@echo "$(BLUE)DPO Continuous Learning Loop - Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Dependencies:$(NC)"
	@echo "  make install              - Install Python dependencies (dev + prod)"
	@echo "  make install-prod         - Install production dependencies only"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  make lint                 - Run ruff (linting) + black (formatting)"
	@echo "  make format               - Auto-format code with black"
	@echo "  make type-check           - Run mypy for type checking"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test                 - Run pytest with coverage report"
	@echo "  make test-unit            - Unit tests only (no integration tests)"
	@echo "  make test-coverage        - Generate HTML coverage report"
	@echo ""
	@echo "$(GREEN)Running Services:$(NC)"
	@echo "  make run-api              - Start FastAPI server (port 8000)"
	@echo "  make run-consumer         - Start Kafka consumer (RUNPOD_MOCK=true by default)"
	@echo "  make run-aggregator       - Run nightly aggregation job (once)"
	@echo "  make run-pipeline         - Run full nightly pipeline (aggregator → trainer → gate → deployer)"
	@echo ""
	@echo "$(GREEN)Local Development:$(NC)"
	@echo "  make docker-up            - Start Kafka, Redis, MLflow (docker-compose)"
	@echo "  make docker-down          - Stop all Docker services"
	@echo "  make seed-data            - Generate mock feedback data"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  make docker-build         - Build API and Trainer Docker images"
	@echo "  make docker-build-api     - Build API image only"
	@echo "  make docker-build-trainer - Build Trainer image only"
	@echo ""
	@echo "$(GREEN)Cleanup:$(NC)"
	@echo "  make clean                - Remove __pycache__, .pytest_cache, coverage reports"
	@echo ""

# ============================================================================
# Setup & Dependencies
# ============================================================================

install:
	@echo "$(BLUE)Installing dependencies (dev + prod)...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-prod:
	@echo "$(BLUE)Installing production dependencies only...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt --no-deps
	@echo "$(GREEN)✓ Production installation complete$(NC)"

# ============================================================================
# Code Quality
# ============================================================================

lint:
	@echo "$(BLUE)Running linter (ruff)...$(NC)"
	ruff check src/ tests/ scripts/ --fix
	@echo "$(GREEN)✓ Linting complete$(NC)"
	@echo ""
	@echo "$(BLUE)Running formatter (black)...$(NC)"
	black src/ tests/ scripts/ --line-length 100
	@echo "$(GREEN)✓ Formatting complete$(NC)"

format:
	@echo "$(BLUE)Auto-formatting with black...$(NC)"
	black src/ tests/ scripts/ --line-length 100
	@echo "$(GREEN)✓ Formatting complete$(NC)"

type-check:
	@echo "$(BLUE)Running type checker (mypy)...$(NC)"
	mypy src/ --ignore-missing-imports --show-error-codes
	@echo "$(GREEN)✓ Type check complete$(NC)"

# ============================================================================
# Testing
# ============================================================================

test:
	@echo "$(BLUE)Running pytest with coverage...$(NC)"
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=75
	@echo ""
	@echo "$(GREEN)✓ Tests complete. Coverage report: htmlcov/index.html$(NC)"

test-unit:
	@echo "$(BLUE)Running unit tests only...$(NC)"
	pytest tests/unit/ -v -m "not integration"
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

test-coverage:
	@echo "$(BLUE)Generating coverage report...$(NC)"
	pytest tests/ --cov=src --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated: htmlcov/index.html$(NC)"

# ============================================================================
# Running Services
# ============================================================================

run-api:
	@echo "$(BLUE)Starting FastAPI server (port 8000)...$(NC)"
	@echo "$(GREEN)Open http://localhost:8000/docs for API documentation$(NC)"
	python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

run-consumer:
	@echo "$(BLUE)Starting Kafka consumer (RUNPOD_MOCK=true)...$(NC)"
	@echo "$(GREEN)Waiting for feedback events from Kafka topic: feedback.events$(NC)"
	RUNPOD_MOCK=true python -m scripts.run_consumer

run-aggregator:
	@echo "$(BLUE)Running nightly aggregation job (one-time)...$(NC)"
	python -m scripts.run_aggregator

run-pipeline:
	@echo "$(BLUE)Running full nightly pipeline...$(NC)"
	@echo "$(GREEN)Chain: Aggregator → Baseline Check → Trainer → Quality Gate → Deployer$(NC)"
	python -m scripts.run_pipeline

# ============================================================================
# Local Development
# ============================================================================

docker-up:
	@echo "$(BLUE)Starting local development stack (Kafka, Redis, MLflow)...$(NC)"
	docker-compose -f docker/docker-compose.dev.yml up -d
	@echo ""
	@echo "$(GREEN)✓ Services started:$(NC)"
	@echo "  - Kafka: localhost:9092"
	@echo "  - Redis: localhost:6379"
	@echo "  - MLflow: http://localhost:5000"
	@echo ""
	@echo "Wait ~10 seconds for services to be healthy before running commands."

docker-down:
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	docker-compose -f docker/docker-compose.dev.yml down
	@echo "$(GREEN)✓ Services stopped$(NC)"

seed-data:
	@echo "$(BLUE)Generating mock feedback data (1000 samples)...$(NC)"
	python scripts/seed_data.py --num-samples 1000 --output /tmp/feedback.jsonl
	@echo "$(GREEN)✓ Mock data generated: /tmp/feedback.jsonl$(NC)"

# ============================================================================
# Docker Images
# ============================================================================

docker-build: docker-build-api docker-build-trainer
	@echo "$(GREEN)✓ All Docker images built$(NC)"

docker-build-api:
	@echo "$(BLUE)Building API Docker image...$(NC)"
	docker build -f docker/Dockerfile.api -t dpo-api:latest -t dpo-api:$(shell git rev-parse --short HEAD) .
	@echo "$(GREEN)✓ API image built: dpo-api:latest$(NC)"

docker-build-trainer:
	@echo "$(BLUE)Building Trainer Docker image...$(NC)"
	docker build -f docker/Dockerfile.trainer -t dpo-trainer:latest -t dpo-trainer:$(shell git rev-parse --short HEAD) .
	@echo "$(GREEN)✓ Trainer image built: dpo-trainer:latest$(NC)"

docker-push: docker-build
	@echo "$(BLUE)Pushing Docker images to registry...$(NC)"
	@echo "Configure DOCKER_REGISTRY in environment (e.g., docker.io/username)"
	docker tag dpo-api:latest ${DOCKER_REGISTRY}/dpo-api:latest
	docker tag dpo-trainer:latest ${DOCKER_REGISTRY}/dpo-trainer:latest
	docker push ${DOCKER_REGISTRY}/dpo-api:latest
	docker push ${DOCKER_REGISTRY}/dpo-trainer:latest
	@echo "$(GREEN)✓ Images pushed$(NC)"

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

.DEFAULT_GOAL := help
