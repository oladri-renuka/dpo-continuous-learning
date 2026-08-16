"""RunPod Serverless API client for GPU training job submission."""

import os
import time
import json
from typing import Dict, Any, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

log = structlog.get_logger(__name__)


class RunPodClientError(Exception):
    """Custom exception for RunPod operations."""
    pass


class RunPodClient:
    """RunPod Serverless API client for submitting and monitoring GPU training jobs."""

    API_BASE_URL = "https://api.runpod.io/v2"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize RunPod client.

        Args:
            api_key: RunPod API key (defaults to RUNPOD_API_KEY env var)

        Raises:
            RunPodClientError: If API key is missing
        """
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise RunPodClientError("RUNPOD_API_KEY environment variable must be set")

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        log.info("RunPod client initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def submit_job(
        self,
        endpoint_id: str,
        input_data: Dict[str, Any],
        timeout: int = 3600,
    ) -> str:
        """
        Submit a training job to RunPod Serverless.

        Args:
            endpoint_id: RunPod endpoint ID
            input_data: Input data for the job (training config, data paths, etc.)
            timeout: Job timeout in seconds

        Returns:
            Job ID

        Raises:
            RunPodClientError: If job submission fails
        """
        try:
            url = f"{self.API_BASE_URL}/{endpoint_id}/run"

            payload = {
                "input": input_data,
            }

            log.info(
                "Submitting RunPod job",
                endpoint_id=endpoint_id,
                url=url,
            )

            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()

            if "id" not in result:
                raise RunPodClientError(f"Invalid RunPod response: {result}")

            job_id = result["id"]
            log.info("RunPod job submitted successfully", job_id=job_id)

            return job_id

        except requests.exceptions.RequestException as e:
            log.error("Failed to submit RunPod job", error=str(e), url=url)
            raise RunPodClientError(f"Failed to submit RunPod job: {str(e)}") from e
        except Exception as e:
            log.error("Unexpected error submitting RunPod job", error=str(e))
            raise RunPodClientError(f"Unexpected error: {str(e)}") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def poll_job(
        self,
        endpoint_id: str,
        job_id: str,
        timeout: int = 3600,
        poll_interval: int = 30,
    ) -> Dict[str, Any]:
        """
        Poll RunPod job status until completion or timeout.

        Args:
            endpoint_id: RunPod endpoint ID
            job_id: Job ID to poll
            timeout: Max polling time in seconds
            poll_interval: Poll interval in seconds

        Returns:
            Job result dict with status and output

        Raises:
            RunPodClientError: If polling fails or job fails
        """
        try:
            url = f"{self.API_BASE_URL}/{endpoint_id}/status/{job_id}"

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    log.debug("Polling RunPod job status", job_id=job_id)
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()

                    result = response.json()

                    # Check job status
                    if result.get("status") == "COMPLETED":
                        log.info("RunPod job completed", job_id=job_id)
                        return result

                    elif result.get("status") == "FAILED":
                        error_msg = result.get("output", {}).get("error", "Unknown error")
                        log.error("RunPod job failed", job_id=job_id, error=error_msg)
                        raise RunPodClientError(f"RunPod job failed: {error_msg}")

                    elif result.get("status") in ["RUNNING", "IN_QUEUE"]:
                        log.debug("Job still running", job_id=job_id, status=result.get("status"))

                    else:
                        log.warning("Unexpected job status", job_id=job_id, status=result.get("status"))

                    # Wait before next poll
                    time.sleep(poll_interval)

                except requests.exceptions.RequestException as e:
                    log.error("Failed to poll RunPod job status", error=str(e))
                    raise RunPodClientError(f"Failed to poll job: {str(e)}") from e

            # Timeout reached
            error_msg = f"RunPod job did not complete within {timeout} seconds"
            log.error(error_msg, job_id=job_id)
            raise RunPodClientError(error_msg)

        except RunPodClientError:
            raise
        except Exception as e:
            log.error("Unexpected error polling RunPod job", error=str(e))
            raise RunPodClientError(f"Unexpected error: {str(e)}") from e

    def submit_and_wait(
        self,
        endpoint_id: str,
        input_data: Dict[str, Any],
        timeout: int = 3600,
        poll_interval: int = 30,
    ) -> Dict[str, Any]:
        """
        Submit a job and wait for completion (convenience method).

        Args:
            endpoint_id: RunPod endpoint ID
            input_data: Input data for the job
            timeout: Max wait time in seconds
            poll_interval: Poll interval in seconds

        Returns:
            Job result dict

        Raises:
            RunPodClientError: If job fails or times out
        """
        job_id = self.submit_job(endpoint_id, input_data, timeout)
        log.info("Waiting for RunPod job completion", job_id=job_id, timeout=timeout)
        return self.poll_job(endpoint_id, job_id, timeout, poll_interval)

    def get_job_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract job output from RunPod result.

        Args:
            result: Result dict from poll_job

        Returns:
            Job output dict (typically contains metrics and artifact paths)
        """
        return result.get("output", {})


def main():
    """Example usage of RunPod client."""
    try:
        client = RunPodClient()

        # Example: submit training job
        config = {
            "train_data": "s3://ml-artifacts/preference-data/2026-08-16/train.jsonl",
            "val_data": "s3://ml-artifacts/preference-data/2026-08-16/val.jsonl",
            "base_model": "meta-llama/Llama-2-7b-hf",
            "lora_rank": 16,
            "learning_rate": 5e-4,
            "epochs": 3,
        }

        # Replace with your actual endpoint ID
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "your-endpoint-id")

        log.info("Submitting training job to RunPod", endpoint_id=endpoint_id)
        result = client.submit_and_wait(
            endpoint_id=endpoint_id,
            input_data=config,
            timeout=7200,  # 2 hours
            poll_interval=30,
        )

        output = client.get_job_output(result)
        log.info("Training job completed", output=output)

    except RunPodClientError as e:
        log.error("RunPod error", error=str(e))
        raise
    except Exception as e:
        log.error("Unexpected error", error=str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
