"""Pydantic schemas for data validation and serialization."""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class DifficultyLevel(str, Enum):
    """Difficulty levels for golden eval examples."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GoldenEvalExample(BaseModel):
    """Single example from the golden evaluation set."""

    id: str = Field(..., description="Unique ID for this example (e.g., 'golden_001')")
    prompt: str = Field(..., description="Input prompt to the model")
    preferred: str = Field(..., description="Preferred (higher quality) response")
    rejected: str = Field(..., description="Rejected (lower quality) response")
    category: str = Field(..., description="Category (e.g., 'education', 'technical', 'reasoning')")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    human_rater: str = Field(..., description="Expert who curated this example")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expected_score: float = Field(
        ..., ge=0.0, le=1.0, description="Human-assigned quality score (0-1)"
    )
    notes: Optional[str] = Field(default=None, description="Optional curation notes")

    class Config:
        use_enum_values = True


class GoldenEvalSet(BaseModel):
    """Container for the golden evaluation set."""

    version: str = Field(..., description="Version identifier (e.g., 'v1', 'v2')")
    examples: List[GoldenEvalExample] = Field(..., description="List of evaluation examples")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Version metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    total_examples: int = Field(..., description="Total number of examples")

    @validator("total_examples")
    def validate_total(cls, v, values):
        if "examples" in values and len(values["examples"]) != v:
            raise ValueError("total_examples must match length of examples list")
        return v

    class Config:
        use_enum_values = True


class FeedbackEvent(BaseModel):
    """User feedback event from Kafka."""

    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Session identifier")
    prompt: str = Field(..., description="User's input prompt")
    sft_response: str = Field(..., description="SFT baseline model response")
    rl_response: str = Field(..., description="RL-aligned model response")
    preferred: str = Field(..., description="Which response user preferred: 'sft' or 'rl'")
    feedback_type: str = Field(..., description="Type: 'thumbs_up' or 'thumbs_down'")
    timestamp: int = Field(..., description="Unix timestamp")
    model_version: str = Field(..., description="Model version at time of feedback")

    @validator("preferred")
    def validate_preferred(cls, v):
        if v not in ("sft_response", "rl_response"):
            raise ValueError("preferred must be 'sft_response' or 'rl_response'")
        return v

    class Config:
        use_enum_values = True


class PreferencePair(BaseModel):
    """Aggregated preference pair for training."""

    prompt: str = Field(..., description="The prompt")
    preferred: str = Field(..., description="Preferred response")
    rejected: str = Field(..., description="Rejected response")
    preference_count: int = Field(default=1, description="Number of times this preference was observed")
    category: Optional[str] = Field(default=None, description="Content category (optional)")

    class Config:
        use_enum_values = True


class QualityGateMetrics(BaseModel):
    """Metrics computed by the quality gate."""

    reward_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Reward model accuracy on golden eval set"
    )
    dpo_win_rate: float = Field(
        ..., ge=0.0, le=1.0, description="DPO win-rate (challenger vs. champion)"
    )
    threshold_acc: float = Field(default=0.72, description="Minimum acceptable accuracy")
    threshold_win_rate: float = Field(default=0.55, description="Minimum acceptable win-rate")
    passed: bool = Field(..., description="Whether quality gate passed all thresholds")
    challenger_version: str = Field(..., description="Version of challenger model")
    champion_version: str = Field(..., description="Version of champion model")
    golden_eval_version: str = Field(..., description="Version of golden eval set used")
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = Field(default=None, description="Error message if validation failed")

    class Config:
        use_enum_values = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MLflow logging."""
        return {
            "reward_accuracy": self.reward_accuracy,
            "dpo_win_rate": self.dpo_win_rate,
            "threshold_acc": self.threshold_acc,
            "threshold_win_rate": self.threshold_win_rate,
            "passed": self.passed,
            "challenger_version": self.challenger_version,
            "champion_version": self.champion_version,
            "golden_eval_version": self.golden_eval_version,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
        }


class TrainingConfig(BaseModel):
    """Configuration for DPO training."""

    base_model: str = Field(default="meta-llama/Llama-2-7b-hf")
    lora_rank: int = Field(default=16)
    lora_alpha: int = Field(default=32)
    lora_dropout: float = Field(default=0.1)
    batch_size: int = Field(default=16)
    learning_rate: float = Field(default=5e-4)
    num_epochs: int = Field(default=3)
    dpo_beta: float = Field(default=0.1, description="KL penalty weight in DPO loss")
    max_prompt_length: int = Field(default=512)
    max_response_length: int = Field(default=512)

    class Config:
        use_enum_values = True


class DeploymentStatus(str, Enum):
    """Status of a deployment."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class DeploymentRecord(BaseModel):
    """Record of a deployment event."""

    version: str = Field(..., description="Model version being deployed")
    status: DeploymentStatus = Field(...)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = Field(default=None)
    reward_accuracy: float = Field(..., description="Accuracy from quality gate")
    dpo_win_rate: float = Field(..., description="Win-rate from quality gate")
    quality_gate_passed: bool = Field(...)
    canary_success_rate: Optional[float] = Field(default=None)
    canary_duration_seconds: Optional[int] = Field(default=None)
    rollback_reason: Optional[str] = Field(default=None)
    notes: Optional[str] = Field(default=None)

    class Config:
        use_enum_values = True
