# DPO Continuous Learning Loop: Architecture Design Document

**Status**: Design Phase 1  
**Date**: 2026-08-16  
**Author**: Staff ML Engineer (Mentor)  
**Target Metrics**: Reward Model Accuracy > 72%, DPO Win-Rate > 55%

---

## Executive Summary

This document defines the architecture for a **production-grade continuous preference learning system** that:

1. **Consumes** real-time user feedback (binary preference pairs) from Kafka
2. **Aggregates** feedback nightly into training datasets
3. **Trains** a reward model + DPO-aligned LLM using QLoRA on RunPod GPUs
4. **Validates** against strict quality gates (metric thresholds are hard stops, no bypass)
5. **Deploys** via Blue/Green strategy with zero-downtime cutover
6. **Monitors** via Prometheus metrics and MLflow experiment tracking

The system is designed to be **self-healing**, **locally testable**, and **failure-resistant**. All external services (Kafka, Redis, RunPod, MLflow) are behind abstracted interfaces to enable offline development with mocks.

---

## 1. System Architecture Overview

### 1.1 High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTINUOUS PREFERENCE LEARNING LOOP                │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │    USER FEEDBACK STREAM (Kafka)         │
                    │  thumbs_up, thumbs_down events         │
                    └──────────────┬──────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                      │
        ┌───────▼────────────┐           ┌────────────▼──────────┐
        │   FastAPI Server   │           │  Kafka Consumer       │
        │  - Chat API        │           │  (Real-time Ingestion)│
        │  - Health/Metrics  │           │  - Dedup (Redis)      │
        │  - Playground      │           │  - Dead-Letter Queue  │
        └──────────┬─────────┘           └────────────┬──────────┘
                   │                                   │
                   │ (Logs events to Kafka)            │
                   │                                   │
                   │                   ┌───────────────▼─────────────┐
                   │                   │  Nightly Aggregation Job    │
                   │                   │  (Cron Trigger)             │
                   │                   │  - Fetch from Redis         │
                   │                   │  - Dedup preference pairs   │
                   │                   │  - Train/Val split (9/1)    │
                   │                   │  - Save to S3               │
                   │                   └───────────────┬─────────────┘
                   │                                   │
                   │                                   │
        ┌──────────┴─────────────────────────────────▼───────────────┐
        │             BASELINE CHECK (scripts/baseline_check.py)      │
        │  - Loads 1000 samples                                       │
        │  - Trains logistic regression (Sentence-BERT)               │
        │  - Validates accuracy > 55% (HARD GATE)                     │
        │  - Exit code 1 if data is intrinsically noisy               │
        └──────────────────────────┬──────────────────────────────────┘
                                   │
        ┌──────────────────────────▼───────────────────────────────┐
        │          RunPod Serverless GPU Job (trainer.py)          │
        │  - Load base LLM (e.g., Llama-7B)                        │
        │  - Train reward model (binary classifier)                │
        │  - Fine-tune LLM with QLoRA + DPO loss                  │
        │  - Save LoRA adapter to S3                              │
        │  - Log metrics to MLflow                                │
        └──────────────────┬───────────────────────────────────────┘
                           │
        ┌──────────────────▼───────────────────────────────────┐
        │        QUALITY GATE (src/core/quality_gate.py)       │
        │  BLOCKING VALIDATION:                               │
        │  ✓ Load Challenger (newly trained) model            │
        │  ✓ Run on 100 golden examples (held-out set)        │
        │  ✓ Compute: Reward Model Accuracy, DPO Win-Rate    │
        │  ✓ Check: Acc > 72% AND Win-Rate > 55%             │
        │  ✗ If FAIL: Raise ModelDegradationError             │
        │           Log "METRIC_FAIL: Got X.X, Expected 72"   │
        │           Exit code 1 (NO BYPASS FLAG)              │
        │  ✓ If PASS: Trigger Blue/Green deployment          │
        └──────────────────┬───────────────────────────────────┘
                           │ [IF PASS]
        ┌──────────────────▼───────────────────────────────────┐
        │       BLUE/GREEN DEPLOYMENT (Zero Downtime)          │
        │  1. Spin up Challenger pod with new LoRA adapter    │
        │  2. Health-check endpoint (/health)                 │
        │  3. Gradual traffic shift (canary: 10% → 50% → 100%)│
        │  4. Monitor metrics for 5 min                        │
        │  5. If degradation detected: Auto-rollback to Blue  │
        │  6. Update active champion pointer                  │
        └──────────────────┬───────────────────────────────────┘
                           │
        ┌──────────────────▼───────────────────────────────────┐
        │         MONITORING & OBSERVABILITY                   │
        │  - Prometheus metrics (/metrics endpoint)            │
        │  - MLflow experiment tracking                        │
        │  - Structured JSON logging (structlog)               │
        │  - Alert thresholds (CloudWatch / Grafana)           │
        └───────────────────────────────────────────────────────┘
```

---

## 2. Data Flow: Step-by-Step

### 2.1 Feedback Collection → Kafka

```
USER INTERACTION:
  Thumbs Up   ─┐
  Thumbs Down ─┼─→ [Event Payload] ─→ Kafka Topic: feedback.events
              │
Event Structure:
{
  "user_id": "user_123",
  "session_id": "sess_456",
  "prompt": "Explain quantum computing",
  "responses": {
    "sft_response": "Quantum computers use...",
    "rl_response": "Quantum computers leverage..."
  },
  "preferred": "rl_response",  # or "sft_response"
  "timestamp": 1692864000,
  "model_version": "v1.2",
  "feedback_type": "thumbs_up"  # or "thumbs_down"
}
```

### 2.2 Real-Time Ingestion (Kafka Consumer)

```
Kafka Consumer (run_consumer.py):
  1. Subscribe to: feedback.events
  2. For each message:
     a. Deserialize JSON
     b. Validate schema (Pydantic)
     c. Check Redis for duplicate (dedup within 5-min window)
     d. If duplicate: Log & discard
     e. If new: Store in Redis (TTL = 24h)
     f. Acknowledge to Kafka
  3. Dead-Letter Queue (DLQ):
     - Invalid schema → feedback.dlq.schema_error
     - Processing error → feedback.dlq.processing_error
     - Manually reviewed daily
```

### 2.3 Nightly Aggregation

```
Cron Trigger (e.g., 02:00 UTC daily):
  1. Connect to Redis
  2. Fetch all feedback from past 24h
  3. Dedup by (prompt, response_pair) hash
  4. Filter:
     - Remove examples < 50 chars (spam)
     - Remove examples > 4096 tokens (truncate)
     - Remove if same user submitted >5% of data (user bias)
  5. Split: Train 90%, Validation 10% (stratified by prompt category)
  6. Save to S3:
     - s3://ml-artifacts/preference-data/{date}/train.jsonl
     - s3://ml-artifacts/preference-data/{date}/val.jsonl
  7. Log stats to MLflow:
     - Total pairs: 5000
     - Duplicate rate: 0.2%
     - Data distribution (by category)
```

### 2.4 Baseline Check (Pre-Training Gate)

```
Script: scripts/baseline_check.py
Input: train.jsonl (1000 samples)

Steps:
  1. Load preference pairs
  2. Embed using Sentence-BERT (universal-sentence-encoder-large)
  3. Extract difference vector: emb_preferred - emb_rejected
  4. Train logistic regression: y = difference_vector → {0, 1}
  5. Evaluate on held-out validation set
  6. Compute accuracy
  7. Log metrics to MLflow:
     - baseline_accuracy (e.g., 0.653)
     - baseline_threshold (0.55)
     - pass_fail (True/False)
     - data_quality_score (computed metric for drift tracking)
     - sample_count (1000)
     - timestamp

OUTPUT:
  ✓ "BASELINE PASSED: Acc=65.3%. Data is learnable. Proceeding to DPO training."
     Logged to MLflow for historical tracking
  ✗ "DATA INTRINSICALLY NOISY: Baseline Acc=48.2%. Fix data or abort."
     Exit code 1 (stops pipeline)
     Also logged to MLflow (with FAILED status) for alerting on data quality degradation

DATA QUALITY DRIFT TRACKING:
  - Over time, view baseline accuracy trend in MLflow
  - If baseline drops below 55%, feedback quality has degraded
  - Typical patterns:
    * Baseline stable (45-65%): Data quality is consistent
    * Baseline dropping (65% → 50%): User feedback becoming more ambiguous
    * Baseline spiking (55% → 80%): Model improvements making feedback clearer
```

### 2.5 Training on RunPod (QLoRA + DPO)

```
RunPod Entrypoint: src/core/trainer.py
Input: 
  - Train/val .jsonl from S3
  - Base model (Llama-2-7B-hf)
  - Config: LoRA rank=16, alpha=32, dropout=0.1

Stage 1: Reward Model Training (Binary Classification)
  Input: Preference pairs (chosen, rejected)
  Architecture: LLM + scalar head
  Loss: Cross-entropy
  Output: reward_model.pth
  Metrics:
    - Training accuracy
    - Validation accuracy → {MUST > 72%}

Stage 2: DPO Fine-Tuning
  Input: Base LLM + preference pairs
  Architecture: QLoRA adapter (rank 16)
  Loss: DPO loss = -log(sigmoid(beta * (policy_log_probs_preferred - policy_log_probs_rejected)))
  Hyperparameters:
    - beta = 0.1 (KL penalty weight)
    - lr = 5e-4
    - epochs = 3
    - batch_size = 16 (per GPU)
  Output: adapter_model.bin (LoRA weights)
  Metrics:
    - DPO loss (decreasing over steps)
    - Preference accuracy (chosen > rejected) → {MUST > 55%}

Checkpointing:
  - Save best checkpoint every 100 steps
  - If OOM: Resume from last checkpoint
  - All artifacts uploaded to S3 immediately after training

Logging:
  - All metrics → MLflow (structured, params + metrics + artifacts)
  - Trainer state (step, loss, learning rate) → structlog → CloudWatch
```

### 2.6 Golden Evaluation Set (Held-Out Validation Data)

```
File: s3://ml-artifacts/golden_eval/v{version}/examples.json
Structure: 100 examples curated by domain experts (stratified by category)

Each example:
{
  "id": "golden_001",
  "prompt": "Explain the difference between supervised and unsupervised learning in the context of modern AI systems...",
  "preferred": "Supervised learning uses labeled data where each input has a corresponding output label. The model learns to map inputs to outputs. Unsupervised learning finds patterns in unlabeled data without predefined targets...",
  "rejected": "Supervised learning is about classifying data. Unsupervised learning is clustering. They are different approaches to machine learning...",
  "category": "education",
  "difficulty": "medium",  # easy, medium, hard (hard = borderline preference)
  "human_rater": "expert_001",
  "created_at": "2026-08-01T10:00:00Z",
  "expected_score": 0.95,  # Human-assigned quality score (0-1)
  "notes": "Preferred response is more comprehensive and accurate"
}

Distribution (100 total):
  - 50 "easy" examples (clear preference signal)
  - 50 "hard" examples (borderline, subtle preference)
  - Stratified by category: education (30%), technical (40%), reasoning (30%)

Curation Process:
  1. Domain experts review 500 real user feedback examples
  2. Select 100 representative examples with clear quality differences
  3. Assign quality scores based on correctness, clarity, helpfulness
  4. Version and store in S3
  5. Never used in training (held-out completely)
  6. Regenerated only when experts deem current set stale (every 3-6 months)

Versioning:
  - v1: Initial 100 examples (2026-08-01)
  - v2: Updated examples with harder cases (2026-11-01)
  - Always load LATEST version from S3
  - Keep all historical versions for reproducibility
  - MLflow logs which version was used for each Quality Gate run

S3 Path Structure:
  s3://ml-artifacts/golden_eval/v1/examples.json
  s3://ml-artifacts/golden_eval/v1/metadata.json  # Version info, timestamp, expert names
  s3://ml-artifacts/golden_eval/v2/examples.json
  ...
```

### 2.7 Quality Gate (Hard Stop)

```
Script: src/core/quality_gate.py
Trigger: After trainer.py completes

Input:
  - Newly trained LoRA adapter (challenger)
  - Current champion model (from S3)
  - Golden eval set (100 curated, held-out examples, latest version from S3)

Execution:
  1. Load challenger model:
     - Base LLM + new LoRA adapter
  2. Load champion model:
     - Base LLM + previous LoRA adapter
  3. Run inference on golden set:
     - For each prompt:
       - Get reward score (challenger) & (champion)
       - Get preference prediction (challenger) & (champion)
  4. Compute metrics:
     - Reward Model Accuracy: 
       correct_predictions / total_predictions
     - DPO Win-Rate: 
       (challenger_preferred_predictions / total_comparisons)
  5. Apply thresholds:
     if accuracy < 0.72 or win_rate < 0.55:
       raise ModelDegradationError(
         f"METRIC_FAIL: Acc={accuracy:.2f} (expected >0.72), "
         f"WinRate={win_rate:.2f} (expected >0.55)"
       )
       log.error("Deployment HALTED due to quality gate failure")
       sys.exit(1)
     else:
       log.info(f"QUALITY GATE PASSED: Acc={accuracy:.3f}, WinRate={win_rate:.3f}")
       return True  # Trigger Blue/Green deployment

No Bypass Flag:
  - There is NO --force flag to override quality gates
  - Pipeline is physically incapable of deploying bad models
  - Only manual intervention (code change) can bypass this
```

### 2.7 Blue/Green Deployment

```
Deployment Flow (Zero Downtime):

State Before:
  - BLUE (production): Champion model v1.2
  - GREEN (staging): Inactive

State After Quality Gate PASS:
  - Spin up GREEN pod with challenger (v1.3 adapter)
  - Run smoke tests: /health endpoint responds, latency < 1000ms
  - Canary traffic shift:
    * 0-60s: 10% traffic → GREEN, 90% → BLUE
    * 60-120s: Monitor success rate, P99 latency
    * If good: 50% → GREEN
    * If degradation: Rollback to BLUE (automatic)
    * 120-300s: Ramp to 100% → GREEN
  - After 5 min of stable metrics:
    * Update champion pointer to GREEN
    * BLUE becomes standby (hot backup)

Rollback Trigger (Automatic):
  - Success rate drops > 5%
  - P99 latency increases > 200ms
  - Any exception spike
  - Manual trigger via `/admin/rollback` endpoint

Zero Downtime Achieved:
  - Load balancer gradually shifts traffic
  - No request is dropped
  - Previous BLUE pod stays alive until GREEN is stable
```

---

## 3. Component Breakdown

### 3.1 Data Ingestion & Storage

| Component | Technology | Purpose | Justification |
|-----------|-----------|---------|---|
| **Message Broker** | Apache Kafka | Real-time feedback stream | High-throughput, partitionable, replay-capable, industry standard for ML feedback loops |
| **Cache Layer** | Redis | Dedup, rate limiting, temporary feedback storage | In-memory speed, TTL support, atomic operations for dedup |
| **Training Data Storage** | S3 (AWS) | Persistence for aggregated datasets | Cheap, versioned, integrates with RunPod |
| **Model Artifacts** | S3 + MLflow Tracking Server | Store LoRA adapters, base models | MLflow provides experiment lineage + model registry |

### 3.2 Core Training Pipeline

| Component | Technology | Purpose | Code Location |
|-----------|-----------|---------|---|
| **Kafka Consumer** | Confluent Kafka Python SDK | Subscribe to feedback stream, handle DLQ | `src/infra/kafka_client.py` |
| **Aggregator** | Pandas + Pydantic | Nightly batch processing | `src/core/aggregator.py` |
| **Trainer (QLoRA + DPO)** | Hugging Face transformers, peft, trl | Model training on GPU | `src/core/trainer.py` (RunPod entrypoint) |
| **Baseline Check** | scikit-learn + Sentence-BERT | Pre-training data quality gate | `scripts/baseline_check.py` |
| **Quality Gate** | Hugging Face, MLflow API | Post-training validation + hard stop | `src/core/quality_gate.py` |

### 3.3 Serving & Monitoring

| Component | Technology | Purpose | Code Location |
|-----------|-----------|---------|---|
| **API Server** | FastAPI + Uvicorn | Chat endpoint, health, metrics | `src/api/app.py` |
| **Model Loading** | vLLM or Ollama | Fast inference with LoRA adapters | `src/infra/model_server.py` |
| **Metrics Export** | Prometheus SDK | `/metrics` endpoint for Grafana | `src/api/app.py` |
| **Structured Logging** | structlog + JSON formatter | CloudWatch/Datadog ingestion | `src/infra/logging_config.py` |
| **Experiment Tracking** | MLflow | Hyperparams, metrics, model lineage | `src/core/quality_gate.py` (logs to MLflow) |

### 3.4 Deployment & Orchestration

| Component | Technology | Purpose | Code Location |
|-----------|-----------|---------|---|
| **GPU Job Submission** | RunPod Serverless API | Spin up training jobs on demand | `src/infra/runpod_client.py` |
| **Blue/Green Deployment** | Kubernetes + Custom Logic | Zero-downtime model swaps | `src/api/deployment.py` |
| **Configuration** | Pydantic Settings + YAML | Environment-aware config | `config/config.yaml` + `src/models/config.py` |

---

## 4. Failure Modes & Recovery Strategies

### 4.1 Kafka is Down

```
Scenario: Kafka broker unavailable for 2 hours
Response:
  1. Kafka Consumer (run_consumer.py):
     - Exponential backoff retry: 1s, 2s, 4s, 8s, 30s (max)
     - After max retries: Log ERROR, keep consumer running (reconnect on next cycle)
     - Buffered events: Temporary disk queue (optional fallback)
  2. User feedback during downtime:
     - FastAPI /feedback endpoint returns 503 (Service Unavailable)
     - Client retries with backoff
     - Data is not lost if client persists locally
  3. Recovery:
     - Once Kafka comes online, consume backlog from committed offset
     - Aggregator detects duplicate dates → merges gracefully
```

### 4.2 RunPod OOM (Out of Memory)

```
Scenario: Training job runs out of VRAM mid-epoch
Response:
  1. RunPod monitoring detects OOM
  2. Job exits with code 137 (SIGKILL)
  3. Recovery logic (trainer.py):
     - Detect last successful checkpoint
     - Reduce batch size by 50%
     - Reduce LoRA rank from 16 to 8
     - Resume training from checkpoint
     - Re-submit to RunPod with new config
  4. If still OOM:
     - Fallback: Use smaller base model (Llama-3B instead of 7B)
     - Log ERROR with recommendation
     - Exit code 1 (halt pipeline, manual intervention needed)
```

### 4.3 Quality Gate Metric Failure

```
Scenario: DPO training finishes, but accuracy = 68% (below 72% threshold)
Response:
  1. Quality gate detects: accuracy < 72%
  2. Log: "METRIC_FAIL: Acc=68.0%, Expected >72%. Deployment HALTED."
  3. Raise ModelDegradationError
  4. Exit code: 1 (HARD STOP - no bypass flag)
  5. Trigger diagnostics (scripts/diagnose.py):
     - Load challenger vs. champion outputs
     - Sample 50 examples, inspect predictions
     - Check for overfitting: Compare train/val loss ratio
     - Generate RCA report: /tmp/rca_report_{timestamp}.json
  6. Alert: PagerDuty / Slack
  7. Manual Investigation Required:
     - Data quality issue? (feedback quality drops)
     - Label noise? (re-review feedback samples)
     - Model capability? (base model too small)
     - Hyperparameter tuning? (adjust beta, lr, epochs)
```

### 4.4 Data Leakage (Train/Val Contamination)

```
Scenario: Same prompt + response pair appears in both train and validation sets
Response:
  1. Aggregator (src/core/aggregator.py):
     - Compute hash of (prompt, response_pair)
     - Strict dedup by hash before split
     - Log duplicate rate
     - If dup rate > 5%: Warn + flag data quality issue
  2. Validation:
     - Post-training: Load train and val sets
     - Compute overlap in prompt hashes
     - If overlap > 1%: Log WARNING, flag as suspicious
  3. Prevention:
     - Use strict train/val split before dedup
     - Shuffle before split (prevent temporal leakage)
```

### 4.5 Model Degradation Detected Post-Deployment

```
Scenario: Canary metrics show success rate drops from 99% to 94% after model swap
Response:
  1. Blue/Green deployment (src/api/deployment.py):
     - Monitor P99 latency, success rate, error rate
     - Threshold: success_rate < 95% OR latency_p99 > 1200ms
     - If triggered: Automatic rollback
  2. Rollback logic:
     - Shift 100% traffic back to BLUE (champion)
     - Spin down GREEN (challenger)
     - Log incident: timestamp, metrics at failure, rollback time
  3. Alert:
     - Slack: "Auto-rollback triggered: Challenger model showed degradation"
     - PagerDuty: P1 incident
  4. Manual review:
     - Inspect champion vs. challenger outputs
     - Check for mode collapse, unsafe behavior, etc.
     - Decide: retrain or revert to older baseline
```

### 4.6 Redis Connection Failure

```
Scenario: Redis is down during aggregation
Response:
  1. Aggregator attempts connection
  2. Circuit breaker triggers after 3 failed attempts (1s timeout each)
  3. Fallback: Aggregator uses disk-based dedup
     - Write feedback to temp SQLite DB
     - Perform dedup with SQL grouping
     - Upload results to S3 (same as normal flow)
  4. Alert: Warn that Redis is down, fallback mode active
  5. Recovery:
     - Once Redis is back, next run syncs with Redis
```

### 4.7 MLflow Tracking Server Down

```
Scenario: MLflow server unavailable during training
Response:
  1. Trainer attempts logging with 3 retries
  2. After retries fail:
     - Graceful degradation: Log locally to JSON file
     - Continue training (don't block on metrics logging)
  3. Save metrics to disk:
     - s3://ml-artifacts/trainer-logs/{run_id}/metrics.json
  4. Recovery:
     - After MLflow comes back online, manual sync of local metrics
     - Next training run re-logs with fresh timestamps
```

### 4.8 Orchestrator Failure (Cron/Airflow Down)

```
Scenario: Kubernetes CronJob or Airflow scheduler is down; pipeline doesn't trigger
Response:
  1. Monitoring detects no pipeline run in 48 hours
  2. Alert: "Orchestrator offline. Last successful run: 2 days ago"
  3. Recovery:
     - On-call engineer restarts orchestrator service
     - Manual trigger: kubectl create job --from=cronjob/dpo-pipeline manual-trigger-{timestamp}
     - Pipeline runs immediately with same logic (no code change needed)
  4. Prevention:
     - Duplicate orchestrator instances (active-passive)
     - Health check endpoint: GET /orchestrator/health
     - Daily validation that pipeline executed successfully
```

---

## 5. Orchestration Layer (The Pipeline Conductor)

### 5.1 Overview

The entire nightly pipeline is orchestrated by a **deterministic, step-by-step chain** that ensures:

1. **Sequential Execution**: Each step must complete before the next begins
2. **Fail-Fast**: Any step failure halts the entire pipeline immediately (no retrying failed steps automatically)
3. **No Manual Intervention for Success**: Happy path runs entirely unattended
4. **Automatic Alerting on Failure**: PagerDuty/Slack notification at first failure point
5. **Full Observability**: Every step's status logged to MLflow + Prometheus

### 5.2 Pipeline Architecture

```
ORCHESTRATOR (Kubernetes CronJob or Apache Airflow)
├─ Triggered: Daily at 02:00 UTC
│
├─ STEP 1: Aggregator
│  ├─ Consumes last 24h of Kafka feedback
│  ├─ Outputs: train.jsonl, val.jsonl → S3
│  ├─ Logs metrics to MLflow:
│  │  ├─ total_pairs
│  │  ├─ dedup_rate
│  │  ├─ data_distribution (by category)
│  │  └─ timestamp
│  ├─ Failure: Exit code 1 → Alert + Stop pipeline
│  └─ Success: Proceed to Step 2
│
├─ STEP 2: Baseline Check (Data Quality Gate)
│  ├─ Loads 1000 samples from train.jsonl
│  ├─ Trains logistic regression classifier
│  ├─ Computes validation accuracy
│  ├─ Logs to MLflow:
│  │  ├─ baseline_accuracy
│  │  ├─ threshold (0.55)
│  │  ├─ pass_fail
│  │  └─ data_quality_score
│  ├─ Failure (acc < 0.55): 
│  │  ├─ Log ERROR: "DATA INTRINSICALLY NOISY: Acc={X}%"
│  │  ├─ Run scripts/diagnose.py --mode data_quality
│  │  ├─ Exit code 1 → Alert + Stop pipeline
│  │  └─ Do NOT proceed to Step 3
│  └─ Success: Proceed to Step 3
│
├─ STEP 3: RunPod Trainer (QLoRA + DPO)
│  ├─ Spins up GPU instance (A100 or equivalent)
│  ├─ Trains reward model + DPO-aligned LLM
│  ├─ Logs to MLflow:
│  │  ├─ reward_model_accuracy
│  │  ├─ dpo_loss (training, validation)
│  │  ├─ preference_accuracy
│  │  ├─ training_time_hours
│  │  └─ hyperparameters (beta, lr, epochs, LoRA rank)
│  ├─ Saves LoRA adapter to S3
│  ├─ Failure (OOM, crash, etc):
│  │  ├─ Retry once with reduced batch size
│  │  ├─ If still fails: Exit code 1 → Alert + Stop pipeline
│  │  └─ Do NOT proceed to Step 4
│  └─ Success: Proceed to Step 4
│
├─ STEP 4: Quality Gate (Metric Validation - HARD STOP)
│  ├─ Loads challenger adapter (newly trained)
│  ├─ Loads champion adapter (current production)
│  ├─ Evaluates on 100 golden examples (held-out)
│  ├─ Computes:
│  │  ├─ Reward model accuracy (challenger)
│  │  └─ DPO win-rate (challenger vs. champion)
│  ├─ Logs to MLflow:
│  │  ├─ reward_accuracy
│  │  ├─ dpo_win_rate
│  │  ├─ threshold_acc (0.72)
│  │  ├─ threshold_win_rate (0.55)
│  │  ├─ pass_fail
│  │  └─ challenger_version
│  ├─ Failure (acc < 0.72 OR win_rate < 0.55):
│  │  ├─ Log ERROR: "QUALITY GATE FAILED: Acc={X}, WinRate={Y}"
│  │  ├─ Raise ModelDegradationError
│  │  ├─ Run scripts/diagnose.py --mode training_quality
│  │  ├─ Exit code 1 (NO BYPASS FLAG) → Alert + Stop pipeline
│  │  └─ Do NOT proceed to Step 5 (model is NOT deployed)
│  └─ Success: Proceed to Step 5
│
└─ STEP 5: Blue/Green Deployer (Canary Rollout)
   ├─ Spin up GREEN pod (challenger model)
   ├─ Health check: GET /health
   ├─ Canary phase: 10% traffic → GREEN, 90% → BLUE
   ├─ Monitor: success_rate, p99_latency, error_rate
   ├─ Logs to MLflow:
   │  ├─ deployment_start_timestamp
   │  ├─ canary_success_rate
   │  ├─ canary_p99_latency
   │  ├─ traffic_shift_progress
   │  └─ deployment_status (pending, in_progress, success, rollback)
   ├─ Failure (success_rate < 95% OR latency > 200ms):
   │  ├─ Automatic rollback to BLUE
   │  ├─ Log ERROR: "CANARY DEGRADATION DETECTED. ROLLING BACK."
   │  ├─ Exit code 1 → Alert (not critical, rollback already happened)
   │  └─ Manual investigation required
   └─ Success:
      ├─ Complete canary: Shift to 100% traffic → GREEN
      ├─ BLUE becomes hot backup (standby)
      ├─ Log SUCCESS: "DEPLOYMENT COMPLETE. Champion updated to {version}"
      ├─ Exit code 0
      └─ Pipeline ends (all steps passed)

MONITORING & ALERTING:
  Every step exit code + metrics logged to:
  ├─ MLflow (experiment: "nightly-pipeline", run per date)
  ├─ Prometheus (gauge: pipeline_step_status)
  ├─ Structlog (JSON to CloudWatch/Datadog)
  └─ PagerDuty + Slack on failure
```

### 5.3 Implementation: Kubernetes CronJob vs. Apache Airflow

#### Option A: Kubernetes CronJob (Simple, Lightweight)

**Best for**: Single-region deployment, simple linear pipeline, DevOps-heavy teams.

```yaml
# k8s/cronjob-dpo-pipeline.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dpo-nightly-pipeline
  namespace: ml-platform
spec:
  schedule: "0 2 * * *"  # 02:00 UTC every day
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 5
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: dpo-pipeline-sa
          containers:
          - name: orchestrator
            image: dpo-orchestrator:latest
            env:
            - name: PIPELINE_MODE
              value: "production"
            - name: RUNPOD_API_KEY
              valueFrom:
                secretKeyRef:
                  name: ml-secrets
                  key: runpod-api-key
            volumeMounts:
            - name: config
              mountPath: /etc/config
          volumes:
          - name: config
            configMap:
              name: dpo-pipeline-config
          restartPolicy: Never
      backoffLimit: 0  # Don't auto-retry; orchestrator handles retries
  # Notification on failure
  postStartHook:
    command: ["/bin/sh", "-c", "send_alert 'Pipeline started'"]
```

**Orchestrator script** (`scripts/run_pipeline.py`):

```python
import sys
import logging
from src.core.aggregator import Aggregator
from src.core.quality_gate import QualityGate, ModelDegradationError
from scripts.baseline_check import run_baseline_check
from src.infra.runpod_client import RunPodClient
from src.api.deployment import BlueGreenDeployer
from src.infra.mlflow_client import MLflowClient

log = logging.getLogger(__name__)

def run_pipeline():
    """Execute the nightly DPO pipeline. Exit code 1 on any failure."""
    
    mlflow = MLflowClient()
    run_id = mlflow.start_run(experiment_name="nightly-pipeline")
    
    try:
        # STEP 1: Aggregator
        log.info("STEP 1: Aggregator - Consuming Kafka feedback...")
        agg = Aggregator()
        agg.aggregate_feedback()
        mlflow.log_params({
            "total_pairs": agg.total_pairs,
            "dedup_rate": agg.dedup_rate,
            "step": "aggregator",
            "status": "completed"
        })
        log.info(f"✓ Aggregator complete: {agg.total_pairs} pairs, dedup rate {agg.dedup_rate:.1%}")
        
        # STEP 2: Baseline Check
        log.info("STEP 2: Baseline Check - Validating data quality...")
        baseline_acc = run_baseline_check()
        mlflow.log_metrics({
            "baseline_accuracy": baseline_acc,
            "baseline_threshold": 0.55,
            "step": "baseline_check"
        })
        if baseline_acc < 0.55:
            log.error(f"DATA INTRINSICALLY NOISY: Baseline Acc={baseline_acc:.2%}. Halting pipeline.")
            mlflow.log_param("baseline_status", "FAILED")
            raise RuntimeError(f"Baseline check failed: {baseline_acc:.2%} < 0.55")
        log.info(f"✓ Baseline check passed: Acc={baseline_acc:.2%}")
        
        # STEP 3: RunPod Trainer
        log.info("STEP 3: RunPod Trainer - Starting GPU training job...")
        runpod = RunPodClient()
        job_result = runpod.submit_training_job(config={
            "train_data": "s3://ml-artifacts/preference-data/{date}/train.jsonl",
            "val_data": "s3://ml-artifacts/preference-data/{date}/val.jsonl",
            "base_model": "meta-llama/Llama-2-7b-hf",
            "lora_rank": 16,
            "batch_size": 16,
            "epochs": 3
        })
        if job_result["status"] != "completed":
            log.error(f"Training job failed: {job_result}")
            raise RuntimeError(f"Training failed: {job_result['error']}")
        log.info(f"✓ Training complete: adapter saved to {job_result['adapter_path']}")
        
        # STEP 4: Quality Gate (HARD STOP)
        log.info("STEP 4: Quality Gate - Validating metrics...")
        gate = QualityGate(
            challenger_adapter_path=job_result["adapter_path"],
            champion_adapter_path="s3://ml-artifacts/models/champion/adapter_model.bin",
            golden_eval_set_path="s3://ml-artifacts/golden_eval/examples.json"
        )
        try:
            gate.validate()  # Raises ModelDegradationError if thresholds not met
            mlflow.log_metrics(gate.metrics)
            log.info(f"✓ Quality gate passed: Acc={gate.metrics['reward_acc']:.3f}, WinRate={gate.metrics['win_rate']:.3f}")
        except ModelDegradationError as e:
            log.error(f"QUALITY GATE FAILED: {str(e)}")
            mlflow.log_param("quality_gate_status", "FAILED")
            raise
        
        # STEP 5: Blue/Green Deployer
        log.info("STEP 5: Blue/Green Deployer - Orchestrating canary rollout...")
        deployer = BlueGreenDeployer()
        deployment_result = deployer.deploy_canary(
            challenger_adapter=job_result["adapter_path"],
            timeout_seconds=300
        )
        if deployment_result["status"] == "rollback":
            log.error(f"Deployment rolled back due to degradation: {deployment_result['reason']}")
            mlflow.log_param("deployment_status", "ROLLED_BACK")
            raise RuntimeError(f"Deployment failed: {deployment_result['reason']}")
        
        log.info(f"✓ Deployment successful: {job_result['adapter_path']} is now CHAMPION")
        mlflow.log_params({
            "deployment_status": "SUCCESS",
            "new_champion": job_result["adapter_path"]
        })
        mlflow.end_run(status="FINISHED")
        
        log.info("=" * 80)
        log.info("✓✓✓ NIGHTLY PIPELINE COMPLETE ✓✓✓")
        log.info("=" * 80)
        return 0
    
    except Exception as e:
        log.error(f"PIPELINE FAILED at step: {str(e)}", exc_info=True)
        mlflow.end_run(status="FAILED")
        # Alert
        send_alert(f"DPO Pipeline failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = run_pipeline()
    sys.exit(exit_code)
```

#### Option B: Apache Airflow (Advanced, Observable)

**Best for**: Multi-region, complex dependencies, data science teams, existing Airflow infrastructure.

```python
# dags/dpo_nightly_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import logging

log = logging.getLogger(__name__)

default_args = {
    "owner": "ml-platform-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email": ["ml-alerts@company.com"],
    "email_on_failure": True,
}

dag = DAG(
    "dpo_nightly_pipeline",
    default_args=default_args,
    description="Continuous DPO alignment pipeline",
    schedule_interval="0 2 * * *",  # 02:00 UTC daily
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["ml", "dpo", "alignment"],
)

# Define tasks
def run_aggregator(**context):
    from src.core.aggregator import Aggregator
    agg = Aggregator()
    agg.aggregate_feedback()
    context["task_instance"].xcom_push(key="aggregation_stats", value={
        "total_pairs": agg.total_pairs,
        "dedup_rate": agg.dedup_rate
    })
    log.info(f"Aggregation complete: {agg.total_pairs} pairs")

def run_baseline(**context):
    from scripts.baseline_check import run_baseline_check
    from src.infra.mlflow_client import MLflowClient
    acc = run_baseline_check()
    if acc < 0.55:
        raise ValueError(f"Baseline failed: {acc:.2%} < 0.55")
    mlflow = MLflowClient()
    mlflow.log_metric("baseline_accuracy", acc)
    log.info(f"Baseline passed: {acc:.2%}")

def run_trainer(**context):
    from src.infra.runpod_client import RunPodClient
    runpod = RunPodClient()
    job = runpod.submit_training_job(config={...})
    context["task_instance"].xcom_push(key="adapter_path", value=job["adapter_path"])
    log.info(f"Training complete: {job['adapter_path']}")

def run_quality_gate(**context):
    from src.core.quality_gate import QualityGate
    adapter_path = context["task_instance"].xcom_pull(task_ids="train", key="adapter_path")
    gate = QualityGate(
        challenger_adapter_path=adapter_path,
        champion_adapter_path="s3://ml-artifacts/models/champion/adapter_model.bin",
        golden_eval_set_path="s3://ml-artifacts/golden_eval/examples.json"
    )
    gate.validate()  # Raises on failure
    log.info(f"Quality gate passed: Acc={gate.metrics['reward_acc']:.3f}")

def run_deployer(**context):
    from src.api.deployment import BlueGreenDeployer
    adapter_path = context["task_instance"].xcom_pull(task_ids="train", key="adapter_path")
    deployer = BlueGreenDeployer()
    result = deployer.deploy_canary(challenger_adapter=adapter_path, timeout_seconds=300)
    if result["status"] == "rollback":
        raise RuntimeError(f"Deployment rolled back: {result['reason']}")
    log.info("Deployment successful")

# Create task nodes
start = DummyOperator(task_id="start", dag=dag)

aggregator = PythonOperator(
    task_id="aggregator",
    python_callable=run_aggregator,
    dag=dag,
)

baseline = PythonOperator(
    task_id="baseline_check",
    python_callable=run_baseline,
    dag=dag,
)

train = PythonOperator(
    task_id="train",
    python_callable=run_trainer,
    dag=dag,
)

quality_gate = PythonOperator(
    task_id="quality_gate",
    python_callable=run_quality_gate,
    dag=dag,
)

deploy = PythonOperator(
    task_id="deploy",
    python_callable=run_deployer,
    dag=dag,
)

end = DummyOperator(
    task_id="end",
    trigger_rule=TriggerRule.ALL_DONE,  # Always run (for logging)
    dag=dag,
)

# Define dependencies (linear execution)
start >> aggregator >> baseline >> train >> quality_gate >> deploy >> end
```

**Airflow Benefits**:
- Task-level retries (not just pipeline-level)
- Dependency visualization in UI
- Historical run logs + SLA tracking
- Sensor support (wait for external conditions)
- Built-in alerting to Slack/PagerDuty

### 5.4 Monitoring the Orchestrator

```python
# src/infra/orchestrator_health.py

class OrchestratorHealth:
    """Monitor orchestrator and pipeline execution health."""
    
    def check_last_run(self):
        """Verify pipeline ran in last 24 hours."""
        mlflow = MLflowClient()
        latest_run = mlflow.get_latest_run(experiment_name="nightly-pipeline")
        
        if latest_run is None:
            raise RuntimeError("No pipeline runs found in MLflow")
        
        hours_since_run = (datetime.now() - latest_run.end_time).total_seconds() / 3600
        if hours_since_run > 24:
            raise RuntimeError(f"Pipeline hasn't run in {hours_since_run:.1f} hours")
        
        return {
            "last_run": latest_run.end_time,
            "status": latest_run.status,
            "duration_seconds": latest_run.duration,
        }
    
    def check_step_health(self, step_name):
        """Check specific step's recent execution."""
        # Returns: pass_rate, avg_duration, last_error
        pass
```

**Endpoint**: `GET /orchestrator/health`

```json
{
  "status": "healthy",
  "last_run": "2026-08-16T02:15:32Z",
  "last_run_status": "success",
  "hours_since_run": 2.5,
  "step_health": {
    "aggregator": {"pass_rate": 1.0, "avg_duration_sec": 45},
    "baseline_check": {"pass_rate": 0.99, "avg_duration_sec": 120},
    "trainer": {"pass_rate": 0.95, "avg_duration_sec": 7200},
    "quality_gate": {"pass_rate": 0.98, "avg_duration_sec": 90},
    "deployer": {"pass_rate": 0.99, "avg_duration_sec": 300}
  }
}
```

### 5.5 Orchestrator Failure Recovery

```bash
# If Kubernetes CronJob is offline:
kubectl get cronjob dpo-nightly-pipeline
# If SUSPEND=true, resume it:
kubectl patch cronjob dpo-nightly-pipeline -p '{"spec":{"suspend":false}}'

# Manually trigger immediately (without waiting for 2 AM):
kubectl create job --from=cronjob/dpo-nightly-pipeline manual-trigger-$(date +%s)

# Monitor status:
kubectl logs -l job-name=manual-trigger-{timestamp} -f

# If Airflow scheduler is down:
systemctl restart airflow-scheduler
# Or trigger DAG manually:
airflow dags trigger dpo_nightly_pipeline
```

---

## 5. Tech Stack Justification

### Why Kafka (not RabbitMQ)?

| Aspect | Kafka | RabbitMQ |
|--------|-------|----------|
| **Throughput** | 1M+ msgs/sec | 100K msgs/sec |
| **Retention** | Configurable (days/weeks) | In-memory only (transient) |
| **Replay** | Full topic replay from offset | No replay (consumed = deleted) |
| **Partitioning** | Native; scales horizontally | Requires shovel plugin |
| **Deduplication** | Idempotent producer, built-in dedup | Manual tracking required |
| **Ecosystem** | Confluent Cloud, AWS MSK, self-hosted | Cloud options limited |

**Decision**: Kafka for its replay capability (critical for re-running aggregation) and partitioning (scale to millions of feedback events).

### Why Redis (not DynamoDB)?

| Aspect | Redis | DynamoDB |
|--------|-------|----------|
| **Latency** | <1ms (in-memory) | 10-30ms (network I/O) |
| **Cost** | $12-50/month for self-hosted | $1-5 per million requests |
| **TTL** | Native TTL support | Requires DynamoDB TTL (eventual) |
| **Dedup Atomicity** | Transactions with WATCH | Conditional writes (slower) |
| **Operational Overhead** | Minimal (managed services available) | AWS-specific |

**Decision**: Redis for sub-millisecond dedup and simplicity. Use AWS ElastiCache if not self-hosted.

### Why MLflow (not Weights & Biases)?

| Aspect | MLflow | W&B |
|--------|--------|-----|
| **Cost** | Free (self-hosted) | $500+/month (paid tiers) |
| **Model Registry** | Built-in versioning | Built-in but slower |
| **Offline Mode** | Works offline, syncs later | Requires online connection |
| **Dependency** | Lightweight | Heavy SDK |
| **Integration** | Hugging Face, Pytorch Lightning | More comprehensive |

**Decision**: MLflow for cost, offline support, and model registry simplicity.

### Why QLoRA + DPO (not Full Fine-Tune)?

| Method | Memory | Cost | Quality | Training Time |
|--------|--------|------|---------|---|
| **QLoRA + DPO** | 8GB VRAM | $0.50/hour | ★★★★☆ | 2 hours |
| **Full Fine-Tune** | 40GB VRAM | $5/hour | ★★★★★ | 2 hours |
| **LoRA Only** | 12GB VRAM | $1/hour | ★★★☆☆ | 1 hour |
| **Prompt Engineering** | 0 (inference only) | $0.01 | ★★☆☆☆ | 1 day (iteration) |

**Decision**: QLoRA + DPO balances cost, quality, and speed. Enables frequent retraining cycles.

### Why RunPod (not Lambda / SageMaker)?

| Service | GPU Availability | Cost | Setup Time | Scaling |
|---------|------------------|------|-----------|---------|
| **RunPod Serverless** | Immediate | $0.25-0.50/hour | <1 min | Auto |
| **AWS Lambda** | Limited GPU support | $0.02/second | Setup complex | Cold starts |
| **SageMaker Training** | Full control | $0.25-1.00/hour | 5 min | Auto |
| **Local GPU** | If available | $0 | N/A | Fixed |

**Decision**: RunPod for ease, cost, and immediate availability without infrastructure overhead.

---

## 6. Quality Gate Design

### 6.1 The Hard Stop Mechanism

```python
# Pseudocode: src/core/quality_gate.py

class ModelDegradationError(Exception):
    """Raised when quality gate thresholds are not met."""
    pass

class QualityGate:
    def __init__(self, challenger_adapter_path, champion_adapter_path, golden_eval_set_path):
        self.challenger = load_model(challenger_adapter_path)
        self.champion = load_model(champion_adapter_path)
        self.golden_eval = load_json(golden_eval_set_path)  # 100 examples
        self.metrics = {}
    
    def validate(self):
        """Run validation. Raise ModelDegradationError if thresholds not met."""
        # 1. Compute reward model accuracy
        reward_acc = self._compute_reward_accuracy()
        
        # 2. Compute DPO win-rate (challenger vs. champion)
        win_rate = self._compute_win_rate()
        
        # 3. Check thresholds (HARD GATES)
        if reward_acc < 0.72:
            raise ModelDegradationError(
                f"Reward Model Accuracy {reward_acc:.2f} < 0.72. DEPLOYMENT HALTED."
            )
        if win_rate < 0.55:
            raise ModelDegradationError(
                f"DPO Win-Rate {win_rate:.2f} < 0.55. DEPLOYMENT HALTED."
            )
        
        # 4. If we reach here, quality gate PASSED
        log.info(f"✓ QUALITY GATE PASSED: Acc={reward_acc:.3f}, WinRate={win_rate:.3f}")
        self.metrics = {"reward_acc": reward_acc, "win_rate": win_rate}
        return True
    
    def _compute_reward_accuracy(self):
        """Reward model predicts preferred response. Accuracy = correct / total."""
        correct = 0
        for example in self.golden_eval:
            pred_reward_preferred = self.challenger.reward_model(example["preferred"])
            pred_reward_rejected = self.challenger.reward_model(example["rejected"])
            if pred_reward_preferred > pred_reward_rejected:
                correct += 1
        return correct / len(self.golden_eval)
    
    def _compute_win_rate(self):
        """DPO win-rate = % of examples where challenger outperforms champion."""
        wins = 0
        for example in self.golden_eval:
            challenger_score = self.challenger(example["prompt"])["score"]
            champion_score = self.champion(example["prompt"])["score"]
            if challenger_score > champion_score:
                wins += 1
        return wins / len(self.golden_eval)
```

### 6.2 Pipeline Exit on Failure

```bash
# In orchestration script (e.g., run_training.sh)

python -m src.core.quality_gate
GATE_EXIT_CODE=$?

if [ $GATE_EXIT_CODE -ne 0 ]; then
    echo "QUALITY GATE FAILED. Pipeline halted. Exit code: $GATE_EXIT_CODE"
    # Trigger alerts
    send_slack_alert("Model degradation detected. Manual investigation required.")
    # Do NOT attempt deployment
    exit 1
else
    echo "Quality gate passed. Proceeding to Blue/Green deployment."
    python -m src.api.deployment trigger_blue_green
fi
```

---

## 7. Deployment Strategy: Blue/Green with Canary

### 7.1 Canary Rollout

```
Timeline (300 seconds total):

Time 0s (Quality gate PASS):
  - Spin up GREEN pod with challenger adapter
  - Health check: GET /health
  - If unhealthy for 10s: Abort deployment, rollback
  
Time 0-60s (Warm-up + Initial Canary):
  - Route 10% traffic to GREEN, 90% to BLUE
  - Monitor: success_rate, p99_latency, error_rate
  - Baseline: success_rate_blue=99.5%, p99_blue=150ms
  
Time 60-120s (Ramp up):
  - If green_success_rate >= 98% AND green_p99 <= 200ms:
      Shift 50% traffic to GREEN, 50% to BLUE
  - Else: Rollback (abort, keep 100% BLUE)
  
Time 120-300s (Full deployment):
  - If metrics still good: Ramp to 100% GREEN
  - Continue monitoring for 180 seconds
  - If degradation detected: Auto-rollback
  
Time 300s (Stable state):
  - GREEN is now CHAMPION (production)
  - BLUE becomes standby (warm backup, no traffic)
  - Announce successful deployment
```

### 7.2 Automatic Rollback Conditions

```
If ANY of these trigger, immediate rollback to BLUE:
  1. Success rate drops > 5 percentage points (e.g., 99% → 94%)
  2. P99 latency increases > 200ms (e.g., 150ms → 380ms)
  3. Error rate spike > 2% of traffic
  4. Out-of-memory or crash in GREEN pod
  5. Manual trigger: POST /admin/rollback
```

---

## 8. Local Development Strategy: Mocking & Offline Testing

### 8.1 Environment-Based Mocking

```yaml
# config/config.yaml

development:
  kafka:
    bootstrap_servers: "localhost:9092"
    mock_enabled: true  # Use mock data generator
  redis:
    host: "localhost"
    port: 6379
    mock_enabled: false  # Use real Redis for dedup testing
  runpod:
    api_key: ""  # Leave empty
    mock_enabled: true  # Print "MOCK: Training..." instead of calling RunPod API
    # Training on CPU with small model
    mock_model_size: "small"  # Llama-160M for fast iteration
  mlflow:
    tracking_uri: "http://localhost:5000"
  
production:
  kafka:
    bootstrap_servers: "prod-kafka-broker.internal:9092"
    mock_enabled: false
  runpod:
    api_key: "${RUNPOD_API_KEY}"  # From environment
    mock_enabled: false
```

### 8.2 Mock Services

```python
# Example: src/infra/runpod_client.py

class RunPodClient:
    def __init__(self, api_key, mock=False):
        self.api_key = api_key
        self.mock = mock
    
    def submit_training_job(self, config):
        if self.mock:
            log.info("MOCK: Submitting training job to RunPod")
            # Simulate training
            job_id = f"mock-job-{uuid.uuid4().hex[:8]}"
            log.info(f"MOCK: Job {job_id} queued")
            # Sleep briefly to simulate GPU compute
            time.sleep(2)
            log.info(f"MOCK: Job {job_id} completed (no-op)")
            return {"job_id": job_id, "status": "completed"}
        else:
            # Real RunPod API call
            response = requests.post(
                "https://api.runpod.io/v2/submit",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=config
            )
            return response.json()
```

### 8.3 Local Test Data

```bash
# Generate mock feedback data for local testing
python scripts/seed_data.py --num-samples 1000 --output /tmp/feedback.jsonl

# Start Kafka + Redis locally
docker-compose -f docker/docker-compose.dev.yml up

# Run consumer with mock data
RUNPOD_MOCK=true python -m src.core.aggregator --data-source /tmp/feedback.jsonl

# Verify pipeline works end-to-end
pytest tests/ -v --cov=src --cov-report=html
```

---

## 9. Monitoring & Observability

### 9.1 Key Metrics

```
Application Metrics:
  - /metrics endpoint exports Prometheus format:
    - http_requests_total{method,endpoint,status}
    - http_request_duration_seconds{method,endpoint}
    - model_inference_duration_seconds{model_version}
    - kafka_consumer_lag
    - redis_dedup_rate

ML Metrics (logged to MLflow):
  - reward_model_accuracy
  - dpo_win_rate
  - dpo_loss (training, validation)
  - preference_accuracy
  - training_time_hours

Deployment Metrics:
  - blue_success_rate
  - green_success_rate
  - canary_shift_progress
  - rollback_count
```

### 9.2 Structured Logging

```json
{
  "timestamp": "2026-08-16T14:32:45.123Z",
  "level": "INFO",
  "logger": "src.core.trainer",
  "event": "epoch_complete",
  "epoch": 2,
  "train_loss": 0.45,
  "val_loss": 0.48,
  "lr": 5e-4,
  "duration_sec": 120.5
}
```

### 9.3 Alerting

```
Alert Rules (Grafana / CloudWatch):
  1. Quality Gate Failure: 
     - Condition: metric_fail_count > 0 in last 1 hour
     - Severity: CRITICAL
     - Action: PagerDuty + Slack #ml-alerts
  
  2. Canary Degradation:
     - Condition: green_success_rate < 95% during rollout
     - Severity: CRITICAL
     - Action: Auto-rollback + Slack
  
  3. Kafka Consumer Lag:
     - Condition: consumer_lag > 10000 messages
     - Severity: WARNING
     - Action: Slack #ops
  
  4. Baseline Check Failure:
     - Condition: baseline_accuracy < 55%
     - Severity: CRITICAL
     - Action: PagerDuty + halt pipeline
```

---

## 10. Security & Compliance

### 10.1 Secrets Management

```
DO NOT hardcode API keys in code.

Use environment variables:
  - RUNPOD_API_KEY
  - KAFKA_SASL_PASSWORD
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
  - MLFLOW_TRACKING_PASSWORD (if password-protected)

Load via Pydantic Settings:
  class Settings(BaseSettings):
      runpod_api_key: str = Field(..., env="RUNPOD_API_KEY")
      kafka_password: str = Field(..., env="KAFKA_SASL_PASSWORD")
      class Config:
          env_file = ".env"
```

### 10.2 Model Artifact Versioning

```
All models stored in S3 with versioning:
  s3://ml-artifacts/models/{model_id}/v{version}/adapter_model.bin
  s3://ml-artifacts/models/{model_id}/v{version}/config.json

Metadata stored in MLflow:
  - Commit hash (code version)
  - Training dataset version
  - Hyperparameters
  - Evaluation metrics
  - Timestamp

Enables:
  - Reproducibility
  - Rollback to any prior version
  - Audit trail
```

---

## 11. Success Criteria

### Phase 1 (Design) ✓
- [x] Architecture diagram (Mermaid)
- [x] Data flow (step-by-step)
- [x] Component breakdown
- [x] Failure modes & recovery (including Orchestrator failures)
- [x] Tech stack justified
- [x] Quality gate design
- [x] Deployment strategy (Blue/Green)
- [x] Local dev strategy (mocks)
- [x] **Orchestration Layer** (Kubernetes CronJob vs. Apache Airflow)
- [x] Baseline Check logs to MLflow for data quality drift tracking
- [x] Sequential pipeline execution with fail-fast semantics

### Phase 2 (Implementation)
- [ ] All code files generated per folder structure
- [ ] Every external service abstracted (interfaces for Kafka, Redis, RunPod, MLflow)
- [ ] Makefile with install, lint, test, run-api, run-consumer
- [ ] Dockerfile.api and Dockerfile.trainer (immutable infrastructure)
- [ ] CI/CD stub (.github/workflows/ci.yml)
- [ ] README.md with quick-start instructions

### Phase 3 (Local Testing)
- [ ] Run consumer with mock data: `RUNPOD_MOCK=true python -m src.core.aggregator`
- [ ] Start API server: `make run-api`
- [ ] Chat endpoint works: POST /v1/chat/completions
- [ ] Metrics exposed: GET /metrics
- [ ] Health check: GET /health
- [ ] All tests pass: `make test`

### Phase 4 (Dashboard)
- [ ] Playground tab (side-by-side responses)
- [ ] Metrics dashboard (7-day history)
- [ ] Deployment history table
- [ ] Dark/light theme support

---

## Appendix: File Checklist

```
dpo-continuous-learning/
├── docs/
│   ├── DESIGN.md                    ← You are here
│   └── API_REFERENCE.md
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── aggregator.py           ← Nightly batch job
│   │   ├── trainer.py              ← RunPod entrypoint
│   │   └── quality_gate.py          ← Hard stop validation
│   ├── infra/
│   │   ├── __init__.py
│   │   ├── kafka_client.py         ← Consumer + DLQ
│   │   ├── redis_client.py         ← Dedup, caching
│   │   ├── runpod_client.py        ← GPU job submission
│   │   ├── model_server.py         ← vLLM + LoRA loading
│   │   └── logging_config.py       ← structlog setup
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                  ← FastAPI server
│   │   ├── routes/
│   │   │   ├── chat.py             ← /chat endpoint
│   │   │   ├── feedback.py         ← /feedback endpoint
│   │   │   └── admin.py            ← /admin/rollback, etc.
│   │   └── deployment.py           ← Blue/Green logic
│   ├── ui/
│   │   ├── app.py                  ← Streamlit/Gradio dashboard
│   │   ├── pages/
│   │   │   ├── playground.py
│   │   │   ├── metrics.py
│   │   │   └── deployment_history.py
│   │   └── components/
│   └── models/
│       ├── __init__.py
│       ├── schemas.py              ← Pydantic models
│       └── config.py               ← Settings
├── tests/
│   ├── conftest.py                 ← Pytest fixtures
│   ├── test_aggregator.py          ← Unit tests
│   ├── test_quality_gate.py
│   ├── test_trainer.py
│   └── test_api.py
├── scripts/
│   ├── baseline_check.py           ← Data quality gate (logs to MLflow)
│   ├── diagnose.py                 ← RCA tool for metric failures
│   ├── seed_data.py                ← Mock data generator
│   ├── run_consumer.py             ← Kafka consumer entrypoint
│   ├── run_pipeline.py             ← ORCHESTRATOR (chains all steps)
│   └── orchestrator_health.py      ← Health check endpoint
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.trainer
│   └── docker-compose.dev.yml
├── config/
│   ├── config.yaml                 ← Dev/prod settings
│   └── logging.yaml                ← Log config
├── .env.example
├── Makefile
├── requirements.txt
├── pyproject.toml
├── pytest.ini
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Next Steps: Phase 2

Once you approve this design, I will generate:

1. **All Python source files** with:
   - Full error handling, retries, circuit breakers
   - Structured logging on every significant action
   - Pydantic validation for all inputs
   - No mock data hardcoded (all via config/environment)

2. **Docker images** for both API and trainer (immutable, code copied in)

3. **Test suite** with >80% coverage of quality_gate.py

4. **Local dev stack** (docker-compose with Kafka, Redis, MLflow)

5. **Dashboard** (Streamlit) with real-time metrics from FastAPI + MLflow

6. **Makefile** with all commands ready to go

Are there any aspects of the design you'd like me to clarify or revise before we proceed?
