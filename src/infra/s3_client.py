"""AWS S3 client with retry logic and error handling."""

import json
import os
import boto3
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

log = structlog.get_logger(__name__)


class S3ClientError(Exception):
    """Custom exception for S3 operations."""
    pass


class MissingCredentialsError(Exception):
    """Raised when AWS credentials are missing."""
    pass


class S3Client:
    """S3 client with automatic retries and error handling (production only)."""

    def __init__(self, bucket: str, region: str = "us-east-1"):
        """
        Initialize S3 client (production only).

        Args:
            bucket: S3 bucket name
            region: AWS region

        Raises:
            MissingCredentialsError: If AWS credentials are not available
            S3ClientError: If S3 client initialization fails
        """
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            raise MissingCredentialsError(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables must be set"
            )

        self.bucket = bucket
        self.region = region

        try:
            self.s3_client = boto3.client("s3", region_name=region)
            log.info("S3Client initialized", bucket=bucket, region=region)
        except Exception as e:
            log.error("Failed to initialize S3 client", error=str(e))
            raise S3ClientError(f"Failed to initialize S3 client: {str(e)}") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download_json(self, key: str) -> Dict[str, Any]:
        """
        Download and parse JSON file from S3.

        Args:
            key: S3 object key (path within bucket)

        Returns:
            Parsed JSON content

        Raises:
            S3ClientError: If download or parsing fails
        """
        try:
            log.debug("Downloading JSON from S3", bucket=self.bucket, key=key)
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            data = json.loads(content)
            log.info("Successfully downloaded JSON from S3", bucket=self.bucket, key=key)
            return data
        except json.JSONDecodeError as e:
            log.error("Failed to parse JSON from S3", key=key, error=str(e))
            raise S3ClientError(f"Invalid JSON at s3://{self.bucket}/{key}: {str(e)}") from e
        except Exception as e:
            log.error("Failed to download from S3", key=key, error=str(e))
            raise S3ClientError(f"Failed to download s3://{self.bucket}/{key}: {str(e)}") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upload_json(self, key: str, data: Dict[str, Any]) -> str:
        """
        Upload JSON data to S3.

        Args:
            key: S3 object key
            data: Data to upload

        Returns:
            S3 URI (s3://bucket/key)

        Raises:
            S3ClientError: If upload fails
        """
        try:
            json_str = json.dumps(data, indent=2, default=str)
            log.debug("Uploading JSON to S3", bucket=self.bucket, key=key)
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json_str.encode("utf-8"),
                ContentType="application/json",
            )
            uri = f"s3://{self.bucket}/{key}"
            log.info("Successfully uploaded JSON to S3", uri=uri)
            return uri
        except Exception as e:
            log.error("Failed to upload to S3", key=key, error=str(e))
            raise S3ClientError(f"Failed to upload to s3://{self.bucket}/{key}: {str(e)}") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def list_objects(self, prefix: str) -> List[str]:
        """
        List all objects under a prefix.

        Args:
            prefix: S3 prefix to list

        Returns:
            List of object keys

        Raises:
            S3ClientError: If listing fails
        """
        try:
            log.debug("Listing objects in S3", bucket=self.bucket, prefix=prefix)
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

            keys = []
            for page in pages:
                if "Contents" in page:
                    keys.extend([obj["Key"] for obj in page["Contents"]])

            log.info("Listed objects from S3", bucket=self.bucket, prefix=prefix, count=len(keys))
            return keys
        except Exception as e:
            log.error("Failed to list objects in S3", prefix=prefix, error=str(e))
            raise S3ClientError(f"Failed to list s3://{self.bucket}/{prefix}: {str(e)}") from e

    def download_file(self, key: str, local_path: str) -> None:
        """
        Download file from S3 to local filesystem.

        Args:
            key: S3 object key
            local_path: Local file path to save to

        Raises:
            S3ClientError: If download fails
        """
        try:
            log.debug("Downloading file from S3", bucket=self.bucket, key=key, local_path=local_path)
            self.s3_client.download_file(self.bucket, key, local_path)
            log.info("Successfully downloaded file from S3", key=key, local_path=local_path)
        except Exception as e:
            log.error("Failed to download file from S3", key=key, error=str(e))
            raise S3ClientError(f"Failed to download s3://{self.bucket}/{key}: {str(e)}") from e
