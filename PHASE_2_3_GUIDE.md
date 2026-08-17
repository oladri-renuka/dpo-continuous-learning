# Phase 2.3: Training Pipeline & Orchestration - Quick Start Guide

## Files Generated

```
src/
├── core/
│   ├── aggregator.py              # Nightly Kafka aggregation (320 lines)
│   └── trainer.py                 # QLoRA + DPO training (310 lines)
├── infra/
│   ├── kafka_client.py            # Kafka consumer with DLQ (280 lines)
│   └── metrics.py                 # Prometheus metrics (added)
scripts/
├── run_pipeline.py                # ORCHESTRATOR (chains all steps) (310 lines)
├── baseline_check.py              # Data quality gate (150 lines)
└── diagnose.py                    # RCA tool for failures (290 lines)
```

**Total: ~1,650 lines of production code**

---

## 🎯 Pipeline Architecture

```
Nightly Cron (02:00 UTC)
    ↓
ORCHESTRATOR (run_pipeline.py)
    ├─ STEP 1: Aggregator
    │  └─ Consume Kafka → train.jsonl + val.jsonl (S3)
    │
    ├─ STEP 2: Baseline Check
    │  └─ Validate data quality (logistic regression)
    │     └─ HARD STOP if accuracy < 55%
    │
    ├─ STEP 3: Trainer
    │  └─ QLoRA + DPO training on RunPod
    │     └─ Saves adapter to S3
    │
    ├─ STEP 4: Quality Gate
    │  └─ Validate metrics (>72% acc, >55% win-rate)
    │     └─ HARD STOP if thresholds not met
    │
    └─ STEP 5: Deployer
       └─ Blue/Green deployment with canary rollout
          └─ SUCCESS: New champion live
```

---

## 📊 Component Breakdown

### 1. **Aggregator** (`src/core/aggregator.py`)

Consumes 24h of user feedback from Kafka and creates training datasets.

```python
aggregator = Aggregator()
train_path, val_path = aggregator.aggregate_feedback()
# Output: s3://ml-artifacts/preference-data/{date}/train.jsonl
#         s3://ml-artifacts/preference-data/{date}/val.jsonl
```

**What it does:**
- ✅ Consume messages from Kafka topic `feedback.events`
- ✅ Deduplicate by (prompt, response_pair) hash
- ✅ Filter: remove spam, truncate long responses
- ✅ Check for user bias
- ✅ Split: 90% train, 10% validation
- ✅ Save to S3 as JSONL
- ✅ Log stats to MLflow

**Stats Tracked:**
- Total raw messages
- Valid messages (after dedup)
- Duplicate rate
- Filtered out
- Final train/val split

### 2. **Trainer** (`src/core/trainer.py`)

Trains QLoRA + DPO fine-tuning on RunPod GPUs.

```python
trainer = DPOTrainer(
    train_data_path="s3://ml-artifacts/.../train.jsonl",
    val_data_path="s3://ml-artifacts/.../val.jsonl"
)
adapter_path = trainer.train()
# Output: s3://ml-artifacts/models/champion/v20260816_120000/adapter_model.bin
```

**What it does:**
- ✅ Load training data from S3
- ✅ Train reward model (binary classifier)
- ✅ DPO fine-tuning with LoRA
- ✅ Save LoRA adapter to S3
- ✅ Log metrics to MLflow (reward_acc, dpo_loss, win_rate)

**Hyperparameters:**
```python
base_model: "meta-llama/Llama-2-7b-hf"
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
learning_rate: 5e-4
num_epochs: 3
dpo_beta: 0.1  # KL penalty weight
```

### 3. **Kafka Consumer** (`src/infra/kafka_client.py`)

High-reliability Kafka consumer with DLQ and retries.

```python
consumer = KafkaConsumer(
    bootstrap_servers="localhost:9092",
    topic="feedback.events",
    mock=True  # Uses mock mode if Kafka unavailable
)
messages = consumer.consume_batch(num_messages=1000)
```

**Features:**
- ✅ Automatic retries (exponential backoff)
- ✅ Dead-letter queue for failed messages
- ✅ Mock mode for local testing
- ✅ Graceful error handling

### 4. **Baseline Check** (`scripts/baseline_check.py`)

Pre-training data quality validation.

```bash
python scripts/baseline_check.py --train-data s3://path/train.jsonl
# Output: "BASELINE CHECK RESULT: Accuracy = 0.65"
# Exit code 1 if accuracy < 0.55 (HARD STOP)
```

**What it does:**
- ✅ Embed examples with Sentence-BERT
- ✅ Train logistic regression classifier
- ✅ Validate accuracy > 55%
- ✅ Raise error if data is too noisy

### 5. **Quality Gate** (already implemented in Phase 2.1)

Hard validation after training.

```python
gate = QualityGate(
    challenger_adapter_path="s3://path/adapter.bin",
    champion_adapter_path="s3://path/champion.bin"
)
gate.validate()  # Raises ModelDegradationError if thresholds not met
```

**Thresholds (HARD STOP):**
- Reward Model Accuracy: ≥ 72%
- DPO Win-Rate: ≥ 55%

### 6. **Orchestrator** (`scripts/run_pipeline.py`)

Chains all steps together with error handling and logging.

```bash
python scripts/run_pipeline.py
# Runs all 5 steps sequentially with proper error handling
# Exit code 0 = success, 1 = failure
```

**Error Handling:**
- ✅ HARD STOP on aggregation failure
- ✅ HARD STOP on baseline check failure (noisy data)
- ✅ HARD STOP on training failure
- ✅ HARD STOP on quality gate failure (metrics)
- ✅ Auto-rollback if deployment degradation detected

### 7. **Diagnostics** (`scripts/diagnose.py`)

RCA tool for debugging failures.

```bash
python scripts/diagnose.py --mode full
# Analyzes:
# - Data quality (baseline accuracy)
# - Training quality (overfitting, gradient norms)
# - Model outputs (mode collapse)
# - Metric drift (degradation over time)
# Outputs: /tmp/rca_report_*.json
```

---

## 🚀 Testing the Pipeline Locally

### Test 1: Just the Aggregator

```bash
python -c "
from src.core.aggregator import Aggregator
agg = Aggregator()
train_path, val_path = agg.aggregate_feedback()
print(f'Train: {train_path}')
print(f'Val: {val_path}')
"
```

Expected output:
```
AGGREGATOR: Starting nightly feedback aggregation
Aggregation stats: 100 pairs, dedup_rate=2%, filter_rate=5%
Train: s3://ml-artifacts/preference-data/2026-08-16/train.jsonl
Val: s3://ml-artifacts/preference-data/2026-08-16/val.jsonl
```

### Test 2: Just the Baseline Check

```bash
python scripts/baseline_check.py --num-samples 100
```

Expected output:
```
BASELINE CHECK: Validating data quality
Baseline accuracy: 0.65
✓ Data is learnable. Proceeding to training.
```

### Test 3: Just the Trainer

```bash
python -c "
from src.core.trainer import DPOTrainer
trainer = DPOTrainer(
    train_data_path='s3://ml-artifacts/dummy/train.jsonl',
    val_data_path='s3://ml-artifacts/dummy/val.jsonl'
)
adapter_path = trainer.train()
print(f'Adapter saved to: {adapter_path}')
"
```

Expected output:
```
DPO TRAINER: Starting QLoRA + DPO training
Reward model training complete: 0.78
DPO fine-tuning complete: loss=0.45, win_rate=0.73
TRAINING COMPLETE
```

### Test 4: Full Pipeline (All 5 Steps)

```bash
python scripts/run_pipeline.py
```

Expected output (full 5-step chain):
```
ORCHESTRATOR: Starting DPO nightly pipeline

STEP 1: AGGREGATOR
  Consumed 100 messages, dedup_rate=2%
  ✓ Aggregation successful

STEP 2: BASELINE CHECK
  Baseline accuracy: 0.65
  ✓ Data is learnable

STEP 3: TRAINER (QLoRA + DPO)
  reward_model_accuracy: 0.78
  dpo_loss: 0.45
  dpo_win_rate: 0.73
  ✓ Training complete

STEP 4: QUALITY GATE (METRIC VALIDATION)
  reward_accuracy: 0.78 > 0.72 ✓
  dpo_win_rate: 0.73 > 0.55 ✓
  ✓ Quality gate passed

STEP 5: DEPLOYER (BLUE/GREEN DEPLOYMENT)
  Spinning up GREEN pod...
  Canary: 10% → 50% → 100% traffic
  ✓ Deployment successful

✓✓✓ NIGHTLY PIPELINE COMPLETE ✓✓✓
Duration: 0.12 hours
New champion: v20260816_120000
```

### Test 5: Diagnostics on Failure

If any step fails, run:

```bash
python scripts/diagnose.py --mode full
# Outputs: /tmp/rca_report_20260816_120000.json
```

The report will contain:
- Data quality issues (baseline accuracy)
- Training issues (overfitting, gradient norms)
- Model output issues (mode collapse)
- Metric drift (degradation over time)
- Specific recommendations for each issue

---

## 📋 Error Scenarios & Recovery

### Scenario 1: Kafka is Down (Aggregator Fails)

```
HARD STOP: Aggregation failed
Exit code: 1
Recovery: Consumer retries 3 times with exponential backoff
```

### Scenario 2: Data is Noisy (Baseline Check Fails)

```
HARD STOP: DATA INTRINSICALLY NOISY (accuracy 48% < 55%)
Exit code: 1
Recovery: Run diagnostics → identify user feedback quality issues → fix UI
```

### Scenario 3: Training OOM (Trainer Fails)

```
HARD STOP: Training failed
Exit code: 1
Recovery: Reduce batch size, LoRA rank, or use smaller model
```

### Scenario 4: Metrics Below Threshold (Quality Gate Fails)

```
HARD STOP: QUALITY GATE FAILED (accuracy 68% < 72%)
Exit code: 1
Recovery: Run diagnostics → adjust hyperparameters → retrain
```

### Scenario 5: Canary Degradation (Deployer Fails)

```
Automatic rollback triggered (success rate dropped)
Exit code: 1
Recovery: Inspect model outputs → identify issue → retrain
```

---

## 🔄 Integration with Cron/Airflow

### Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dpo-nightly-pipeline
spec:
  schedule: "0 2 * * *"  # 02:00 UTC daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: orchestrator
            image: dpo-orchestrator:latest
            command: ["python", "scripts/run_pipeline.py"]
          restartPolicy: Never
```

### Apache Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG('dpo_nightly_pipeline', schedule_interval='0 2 * * *')

def run_orchestrator():
    from scripts.run_pipeline import PipelineOrchestrator
    orchestrator = PipelineOrchestrator()
    return orchestrator.run()

task = PythonOperator(
    task_id='dpo_pipeline',
    python_callable=run_orchestrator,
    dag=dag
)
```

---

## 📊 Key Metrics Logged to MLflow

Each step logs metrics that can be viewed in MLflow UI:

**Aggregator:**
- total_raw_messages
- valid_messages
- dedup_rate
- filter_rate
- final_pairs

**Trainer:**
- reward_model_accuracy
- dpo_loss
- dpo_win_rate
- training_time_hours

**Quality Gate:**
- reward_accuracy
- dpo_win_rate
- threshold checks

**Deployer:**
- deployment_status
- canary_success_rate
- traffic_shift_progress

---

## ✅ Success Criteria

Pipeline succeeds (exit code 0) when:

1. ✓ Aggregator produces train/val datasets
2. ✓ Baseline check passes (acc > 55%)
3. ✓ Trainer completes successfully
4. ✓ Quality gate passes (acc > 72%, win-rate > 55%)
5. ✓ Deployer completes canary rollout
6. ✓ New champion is live

Pipeline fails (exit code 1) if ANY step fails.

---

## 🎯 Next Steps

After Phase 2.3, we'll proceed to:

- **Phase 3**: Integration testing with docker-compose (Kafka, Redis, MLflow)
- **Phase 4**: Streamlit dashboard (playground, metrics, deployment history)
- **Phase 5**: Production deployment playbook (Kubernetes manifests, CI/CD)

---

## Terminal Commands Cheat Sheet

```bash
# Run aggregator only
python -m src.core.aggregator

# Run baseline check
python scripts/baseline_check.py --num-samples 1000

# Run trainer only
python -m src.core.trainer

# Run quality gate (already tested in Phase 2.1)
python -m src.core.quality_gate --s3-mock

# Run FULL PIPELINE
python scripts/run_pipeline.py

# Run diagnostics on failure
python scripts/diagnose.py --mode full

# Watch MLflow UI for metrics
mlflow ui --host 0.0.0.0 --port 5000
# Open: http://localhost:5000
```

---

**Phase 2.3 Complete!** All training pipeline components are ready. 🚀
