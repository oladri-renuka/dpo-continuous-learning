"""Configuration management using Pydantic Settings (production only)."""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables or .env file."""

    # ========================================================================
    # API Configuration
    # ========================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    admin_api_key: str = "your-super-secret-admin-key"

    # ========================================================================
    # Kafka Configuration (Production)
    # ========================================================================
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_feedback: str = "feedback.events"

    # ========================================================================
    # RunPod Configuration (Production)
    # ========================================================================
    runpod_api_key: Optional[str] = None

    # ========================================================================
    # AWS S3 Configuration (Production)
    # ========================================================================
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    s3_bucket: str = "ml-artifacts"

    # ========================================================================
    # MLflow Configuration (Production)
    # ========================================================================
    mlflow_tracking_uri: str = "http://localhost:5000"

    # ========================================================================
    # Model Configuration
    # ========================================================================
    base_model_path: str = "meta-llama/Llama-2-7b-hf"
    champion_adapter_path: str = "s3://ml-artifacts/models/champion/adapter_model.bin"
    golden_eval_path: str = "s3://ml-artifacts/golden_eval/v1/examples.json"

    # ========================================================================
    # Feature Flags
    # ========================================================================
    cors_allow_origins: list = ["*"]
    environment: str = "production"  # development, staging, production

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings (singleton pattern)."""
    return Settings()
