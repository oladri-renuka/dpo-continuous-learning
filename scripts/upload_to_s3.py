"""Upload training data to AWS S3."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from src.infra.s3_client import S3Client, S3ClientError

log = structlog.get_logger(__name__)


def upload_training_data(
    train_data_path: str = "./data/train.jsonl",
    val_data_path: str = "./data/val.jsonl",
    bucket: str = "dpo-ml-artifacts",
    prefix: str = "preference-data/2026-08-16",
) -> tuple:
    """
    Upload training data files to S3.

    Args:
        train_data_path: Local path to training JSONL
        val_data_path: Local path to validation JSONL
        bucket: S3 bucket name
        prefix: S3 path prefix

    Returns:
        Tuple of (train_s3_path, val_s3_path)

    Raises:
        S3ClientError: If upload fails
    """
    try:
        # Verify files exist
        train_path = Path(train_data_path)
        val_path = Path(val_data_path)

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_data_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation data not found: {val_data_path}")

        log.info(f"Found training data: {train_path} ({train_path.stat().st_size} bytes)")
        log.info(f"Found validation data: {val_path} ({val_path.stat().st_size} bytes)")

        # Initialize S3 client
        log.info(f"Connecting to S3 bucket: {bucket}")
        s3_client = S3Client(bucket=bucket)

        # Upload training data
        log.info("Uploading training data to S3...")
        with open(train_path, "r") as f:
            train_data = {"examples": []}
            for line in f:
                import json
                example = json.loads(line.strip())
                train_data["examples"].append(example)

        train_key = f"{prefix}/train.jsonl"
        train_s3_path = s3_client.upload_json(train_key, train_data)
        log.info(f"✓ Uploaded training data to {train_s3_path}")

        # Upload validation data
        log.info("Uploading validation data to S3...")
        with open(val_path, "r") as f:
            val_data = {"examples": []}
            for line in f:
                import json
                example = json.loads(line.strip())
                val_data["examples"].append(example)

        val_key = f"{prefix}/val.jsonl"
        val_s3_path = s3_client.upload_json(val_key, val_data)
        log.info(f"✓ Uploaded validation data to {val_s3_path}")

        return train_s3_path, val_s3_path

    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
        raise
    except S3ClientError as e:
        log.error(f"S3 upload failed: {e}")
        raise
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        raise


def main():
    """Upload training data to S3."""
    try:
        from src.infra import setup_logging
        setup_logging(environment="production")
    except ImportError:
        pass

    # Parse arguments
    train_data_path = sys.argv[1] if len(sys.argv) > 1 else "./data/train.jsonl"
    val_data_path = sys.argv[2] if len(sys.argv) > 2 else "./data/val.jsonl"
    bucket = sys.argv[3] if len(sys.argv) > 3 else "dpo-ml-artifacts"
    prefix = sys.argv[4] if len(sys.argv) > 4 else "preference-data/2026-08-16"

    train_s3_path, val_s3_path = upload_training_data(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        bucket=bucket,
        prefix=prefix,
    )

    print("\n" + "=" * 80)
    print("✓ Data uploaded to S3 successfully!")
    print("=" * 80)
    print(f"\nUse these paths in the trainer:")
    print(f"  --train_data {train_s3_path}")
    print(f"  --val_data {val_s3_path}")
    print("\nOr run:")
    print(f"  python -m src.core.trainer {train_s3_path} {val_s3_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
