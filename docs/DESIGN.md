# System Design

## Overview

Feedback-driven LLM alignment pipeline:
1. Collect preferences via Redis pub/sub
2. Dedup with SHA256 + Redis TTL (24h)
3. Aggregate nightly into train/val splits
4. Train LoRA adapters with DPO on Llama-2-7B
5. Validate via quality gates
6. Serve via FastAPI with automatic reloading

## Data Flow

```
User Feedback
     ↓
Redis Pub/Sub (feedback_events channel)
     ↓
Aggregator (src/core/aggregator.py)
  • Subscribe to channel
  • SHA256 dedup with Redis TTL
  • Write to data/raw/feedback_YYYY-MM-DD.jsonl
  • Trigger orchestrator at 500-message threshold
     ↓
Orchestrator (scripts/run_pipeline.py)
  • Read raw feedback files
  • Dedup by (prompt, user_id)
  • Split 90/10 train/val
  • Upload to MinIO S3
  • Submit training job
     ↓
Trainer (src/core/trainer.py)
  • Load Llama-2-7B from HuggingFace
  • Apply QLoRA (rank=16, alpha=32, dropout=0.05)
  • Train with DPO loss
  • Extract training loss and win-rate metrics
  • Save adapter via save_pretrained()
     ↓
Quality Gate (src/core/quality_gate.py)
  • Load challenger adapter
  • Evaluate on held-out test set
  • Check: Accuracy > 72%, Win-Rate > 55%
  • Hard stop if thresholds not met (exit code 1)
     ↓
Model Serving (src/api/app.py)
  • Load champion adapter on startup
  • Poll for champion_pointer.json changes (60s interval)
  • Auto-reload when new model available
  • Serve inference on /predict endpoint
```

## Components

### Aggregator (src/core/aggregator.py)

Real-time consumer for Redis pub/sub channel `feedback_events`.

Deduplication via SHA256:
- Hash = SHA256({prompt, chosen, rejected, user_id})
- Check Redis for duplicate
- Store hash in Redis with 24h TTL
- Skip duplicate, write new feedback to disk

Writes to `data/raw/feedback_YYYY-MM-DD.jsonl` (one line per message).

Triggers orchestrator when message count >= 500.

### Orchestrator (scripts/run_pipeline.py)

End-to-end pipeline coordinator:

1. Read raw feedback from latest file in `data/raw/`
2. Dedup by (prompt, user_id)
3. Split 90/10 into train/val
4. Upload to S3: `train/{date}/train.jsonl`, `train/{date}/val.jsonl`
5. Submit training job:
   - Local: subprocess with src/core/trainer
   - RunPod: API call (if RUNPOD_API_KEY set)
6. Wait for training completion
7. Download adapter from S3
8. Run quality gate validation
9. Deploy as champion (upload to models/champion/)
10. Archive processed data

### Trainer (src/core/trainer.py)

Runs on GPU (local or RunPod).

- Loads Llama-2-7B from meta-llama/Llama-2-7b-hf
- Applies QLoRA with PEFT
- Creates DPOTrainer from TRL
- Trains on provided datasets
- Logs metrics to stdout (captured by orchestrator)
- Saves adapter via `model.save_pretrained()`

Hyperparameters:
- LoRA rank: 16
- LoRA alpha: 32
- Dropout: 0.05
- LR: 1e-4
- Beta (KL weight): 0.05
- Max grad norm: 1.0
- Batch size: 4
- Epochs: 3

### Quality Gate (src/core/quality_gate.py)

Runs after training completes.

Loads challenger adapter and base model. Runs inference on held-out test set.

Computes:
- Accuracy: % of examples where preferred_score > rejected_score
- Win-rate: % of examples where challenger performs better than champion

Thresholds:
- Accuracy >= 0.72 (72%)
- Win-rate >= 0.55 (55%)

If both thresholds met: passes, triggers deployment.
If either fails: raises exception, exit code 1, halts pipeline.

### API Server (src/api/app.py)

FastAPI server for inference.

Endpoints:
- `/health` - returns model status, adapter path, last reload time
- `/predict` - runs inference with loaded adapter
- `/feedback` - accepts preference data (publishes to Redis)

Loads champion model on startup. Polls `outputs/champion_pointer.json` every 60s. Reloads model when pointer changes.

### S3 Client (src/infra/s3_client.py)

Supports both MinIO (local) and AWS S3.

Auto-detects from environment:
- MinIO: S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY
- AWS: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

Methods:
- upload_json(key, data)
- download_json(key)
- upload_file(local_path, key)
- download_file(key, local_path)
- list_objects(prefix)

## Failure Modes

### Redis Down

Aggregator fails to connect. Logs error, raises exception. Retry manually or restart Redis.

### S3/MinIO Down

Orchestrator fails during upload/download. Logs error, exit code 1. Retry when S3 is back online.

### Training OOM

GPU job crashes with SIGKILL. Orchestrator catches failure. Can retry with reduced batch size.

### Quality Gate Failure

Metrics below threshold. Logged as error, exit code 1. No deployment happens. Manual investigation required.

### Model Serving

If champion_pointer.json missing, API loads baseline model. Auto-reloads when pointer file appears.

## Configuration

Environment variables:

```
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL_SECONDS=86400

S3_ENDPOINT_URL=http://localhost:9000  # MinIO
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin123
S3_BUCKET=ml-artifacts

PIPELINE_FEEDBACK_THRESHOLD=500        # Aggregator trigger
PIPELINE_RUNPOD_POLL_INTERVAL=30       # Seconds
PIPELINE_MODEL_RELOAD_INTERVAL=60      # Seconds

TRAINING_BATCH_SIZE=4
TRAINING_EPOCHS=3
TRAINING_LEARNING_RATE=1e-4
TRAINING_BETA=0.05
TRAINING_MAX_GRAD_NORM=1.0

DATA_RAW_DIR=./data/raw
DATA_PROCESSED_DIR=./data/processed
OUTPUT_DIR=./outputs
```

## Local Development

Start infrastructure:
```bash
docker-compose up -d
```

Run aggregator:
```bash
python -m src.core.aggregator
```

Send test feedback:
```bash
python scripts/test_feedback.py
```

Run orchestrator:
```bash
python scripts/run_pipeline.py
```

Start API:
```bash
python -m src.api.app
curl http://localhost:8000/health
```

## Testing

```bash
pytest tests/ -v
```

Integration tests verify:
- Feedback ingestion via Redis
- Aggregator dedup logic
- Orchestrator pipeline steps
- Quality gate validation
- API endpoints

