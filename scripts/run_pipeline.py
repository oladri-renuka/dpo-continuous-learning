"""Orchestrator: Aggregate feedback → Upload to S3 → Submit RunPod job → Quality Gate → Deploy."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import structlog
from dotenv import load_dotenv

log = structlog.get_logger(__name__)

# Load environment
load_dotenv()

DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "./data/raw"))
DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "./data/processed"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))

S3_BUCKET = os.getenv("S3_BUCKET", "ml-artifacts")
RUNPOD_POLL_INTERVAL = int(os.getenv("PIPELINE_RUNPOD_POLL_INTERVAL", "30"))


class Orchestrator:
    """End-to-end training pipeline orchestrator."""

    def __init__(self):
        self.data_raw_dir = DATA_RAW_DIR
        self.data_processed_dir = DATA_PROCESSED_DIR
        self.output_dir = OUTPUT_DIR

        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Import at runtime to avoid hard dependency
        try:
            from src.infra.s3_client import S3Client
            from src.infra.runpod_client import RunPodClient

            self.s3_client = S3Client(bucket=S3_BUCKET)
            self.runpod_client = RunPodClient()
        except Exception as e:
            log.error(f"Failed to initialize clients: {str(e)}")
            raise

        log.info("Orchestrator initialized")

    def _get_latest_raw_file(self) -> Path:
        """Get the most recent raw feedback file."""
        files = sorted(self.data_raw_dir.glob("feedback_*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No feedback files in {self.data_raw_dir}")

        latest = files[-1]
        log.info(f"Found raw file: {latest.name}")
        return latest

    def _read_feedback_file(self, filepath: Path) -> list:
        """Read and parse JSONL feedback file."""
        examples = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        log.info(f"Loaded {len(examples)} examples from {filepath.name}")
        return examples

    def _dedup_and_split(self, examples: list, train_ratio: float = 0.9) -> tuple:
        """Dedup, filter, and split into train/val sets."""
        # Dedup by prompt+user
        seen = set()
        deduped = []

        for ex in examples:
            key = (ex.get("prompt", ""), ex.get("user_id", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(ex)

        log.info(f"Deduplicated: {len(examples)} → {len(deduped)} examples")

        # Split
        split_idx = int(len(deduped) * train_ratio)
        train_examples = deduped[:split_idx]
        val_examples = deduped[split_idx:]

        log.info(f"Split: {len(train_examples)} train, {len(val_examples)} val")

        return train_examples, val_examples

    def _upload_dataset_to_s3(
        self, train_examples: list, val_examples: list
    ) -> tuple:
        """Upload train and val datasets to S3."""
        timestamp = datetime.now().strftime("%Y-%m-%d")

        # Prepare data
        train_data = {"examples": train_examples}
        val_data = {"examples": val_examples}

        # Upload
        train_key = f"train/{timestamp}/train.jsonl"
        val_key = f"train/{timestamp}/val.jsonl"

        try:
            train_s3_path = self.s3_client.upload_json(train_key, train_data)
            val_s3_path = self.s3_client.upload_json(val_key, val_data)

            log.info(f"✓ Uploaded train data: {train_s3_path}")
            log.info(f"✓ Uploaded val data: {val_s3_path}")

            return train_s3_path, val_s3_path
        except Exception as e:
            log.error(f"Failed to upload datasets to S3: {str(e)}")
            raise

    def _submit_training_job(self, train_s3_path: str, val_s3_path: str) -> str:
        """Submit training job to RunPod."""
        log.info("=" * 80)
        log.info("Submitting training job to RunPod...")
        log.info("=" * 80)

        command = f"python -m src.core.trainer {train_s3_path} {val_s3_path}"

        try:
            job_id = self.runpod_client.submit_training_job(
                command=command,
                input_data={"train_data": train_s3_path, "val_data": val_s3_path},
            )
            log.info(f"✓ Job submitted: {job_id}")
            return job_id
        except Exception as e:
            log.error(f"Failed to submit training job: {str(e)}")
            raise

    def _wait_for_training(self, job_id: str, timeout_seconds: int = 3600) -> dict:
        """Poll RunPod API until job completes."""
        log.info(f"Polling RunPod job {job_id} every {RUNPOD_POLL_INTERVAL}s...")

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time

            # Timeout check
            if elapsed > timeout_seconds:
                log.error(f"Job timed out after {elapsed:.0f}s")
                raise TimeoutError(f"Training job {job_id} timed out")

            try:
                # Poll job status
                result = self.runpod_client.get_job_status(job_id)

                if result["status"] == "COMPLETED":
                    log.info(f"✓ Job completed successfully")
                    return result

                elif result["status"] == "FAILED":
                    log.error(f"✗ Job failed: {result.get('error', 'Unknown error')}")
                    raise Exception(f"Training job failed: {result}")

                else:
                    log.debug(f"Status: {result['status']} (elapsed: {elapsed:.0f}s)")

            except Exception as e:
                log.error(f"Error polling job status: {str(e)}")

            # Wait before next poll
            time.sleep(RUNPOD_POLL_INTERVAL)

    def _download_adapter(self, job_id: str) -> Path:
        """Download trained adapter from S3 to local disk."""
        log.info(f"Downloading adapter from S3...")

        try:
            # Expected path: s3://bucket/models/job_id/adapter_model.safetensors
            adapter_key = f"models/{job_id}/adapter_model.safetensors"
            config_key = f"models/{job_id}/adapter_config.json"

            # Download both files
            adapter_path = self.output_dir / "latest"
            adapter_path.mkdir(exist_ok=True)

            self.s3_client.download_file(adapter_key, str(adapter_path / "adapter_model.safetensors"))
            self.s3_client.download_file(config_key, str(adapter_path / "adapter_config.json"))

            log.info(f"✓ Adapter downloaded to {adapter_path}")
            return adapter_path

        except Exception as e:
            log.error(f"Failed to download adapter: {str(e)}")
            raise

    def _run_quality_gate(self, adapter_path: Path) -> bool:
        """Run quality gate on challenger adapter."""
        log.info("=" * 80)
        log.info("Running Quality Gate...")
        log.info("=" * 80)

        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.core.quality_gate",
                    "--challenger",
                    str(adapter_path),
                    "--champion",
                    "baseline",
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
            )

            if result.returncode == 0:
                log.info("✓ Quality gate PASSED")
                return True
            else:
                log.error("✗ Quality gate FAILED")
                log.error(f"Output: {result.stdout}")
                log.error(f"Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            log.error("Quality gate timed out")
            return False
        except Exception as e:
            log.error(f"Failed to run quality gate: {str(e)}")
            return False

    def _deploy_champion(self, adapter_path: Path, job_id: str) -> None:
        """Upload adapter to champion location on S3."""
        log.info("=" * 80)
        log.info("Deploying as champion model...")
        log.info("=" * 80)

        try:
            # Upload adapter files to champion location
            champion_key_model = f"models/champion/adapter_model.safetensors"
            champion_key_config = f"models/champion/adapter_config.json"

            adapter_model_path = adapter_path / "adapter_model.safetensors"
            adapter_config_path = adapter_path / "adapter_config.json"

            self.s3_client.upload_file(str(adapter_model_path), champion_key_model)
            self.s3_client.upload_file(str(adapter_config_path), champion_key_config)

            # Update champion pointer
            champion_pointer = {
                "timestamp": datetime.now().isoformat(),
                "job_id": job_id,
                "s3_path": f"s3://{S3_BUCKET}/models/champion/",
                "status": "CHAMPION",
            }

            pointer_path = self.output_dir / "champion_pointer.json"
            with open(pointer_path, "w") as f:
                json.dump(champion_pointer, f, indent=2)

            log.info(f"✓ Deployed as champion")
            log.info(f"  S3 path: s3://{S3_BUCKET}/models/champion/")
            log.info(f"  Pointer: {pointer_path}")

        except Exception as e:
            log.error(f"Failed to deploy champion: {str(e)}")
            raise

    def _archive_raw_data(self, raw_file: Path) -> None:
        """Move processed raw file to archive."""
        try:
            archive_path = self.data_processed_dir / raw_file.name
            raw_file.rename(archive_path)
            log.info(f"✓ Archived raw file: {archive_path.name}")
        except Exception as e:
            log.error(f"Failed to archive raw file: {str(e)}")

    def run(self) -> None:
        """Execute full pipeline."""
        log.info("=" * 80)
        log.info("ORCHESTRATOR: Starting end-to-end pipeline")
        log.info("=" * 80)

        try:
            # Step 1: Read raw feedback
            log.info("STEP 1: Reading raw feedback...")
            raw_file = self._get_latest_raw_file()
            examples = self._read_feedback_file(raw_file)

            # Step 2: Dedup and split
            log.info("STEP 2: Deduplicating and splitting...")
            train_examples, val_examples = self._dedup_and_split(examples)

            # Step 3: Upload to S3
            log.info("STEP 3: Uploading datasets to S3...")
            train_s3_path, val_s3_path = self._upload_dataset_to_s3(
                train_examples, val_examples
            )

            # Step 4: Submit training job
            log.info("STEP 4: Submitting to RunPod...")
            job_id = self._submit_training_job(train_s3_path, val_s3_path)

            # Step 5: Wait for training
            log.info("STEP 5: Waiting for training to complete...")
            job_result = self._wait_for_training(job_id)

            # Step 6: Download adapter
            log.info("STEP 6: Downloading adapter...")
            adapter_path = self._download_adapter(job_id)

            # Step 7: Run quality gate
            log.info("STEP 7: Running quality gate...")
            if not self._run_quality_gate(adapter_path):
                log.error("Quality gate failed. Aborting deployment.")
                return

            # Step 8: Deploy champion
            log.info("STEP 8: Deploying champion model...")
            self._deploy_champion(adapter_path, job_id)

            # Step 9: Archive raw data
            log.info("STEP 9: Archiving raw data...")
            self._archive_raw_data(raw_file)

            log.info("=" * 80)
            log.info("✓ PIPELINE COMPLETE - Model deployed as champion")
            log.info("=" * 80)

        except Exception as e:
            log.error(f"Pipeline failed: {str(e)}", exc_info=True)
            sys.exit(1)


def main():
    """Entry point."""
    try:
        from src.infra import setup_logging

        setup_logging(environment="production")
    except ImportError:
        pass

    orchestrator = Orchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
