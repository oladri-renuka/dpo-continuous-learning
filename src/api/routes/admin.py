"""Admin endpoints for deployment control."""

from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException
import structlog

from src.infra import get_logger
from src.models.config import get_settings

log = get_logger(__name__)
router = APIRouter()

# ============================================================================
# Admin State (stub)
# ============================================================================

_deployment_history = []
_current_champion = {
    "version": "v1.0",
    "adapter_path": "s3://ml-artifacts/models/champion/adapter_model.bin",
    "deployed_at": datetime.utcnow().isoformat(),
    "reward_accuracy": 0.82,
    "dpo_win_rate": 0.73,
}


# ============================================================================
# Helper Functions
# ============================================================================


def validate_admin_key(x_admin_key: Optional[str]) -> bool:
    """Validate admin API key from request header."""
    settings = get_settings()
    if not x_admin_key or x_admin_key != settings.admin_api_key:
        return False
    return True


# ============================================================================
# Admin Endpoints
# ============================================================================


@router.post("/rollback")
async def trigger_rollback(x_admin_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Trigger immediate rollback to previous champion (BLUE).

    Requires X-Admin-Key header with correct API key.
    """
    if not validate_admin_key(x_admin_key):
        log.warning("Unauthorized rollback attempt", x_admin_key_provided=(x_admin_key is not None))
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing admin key")

    try:
        log.warning(
            "ROLLBACK TRIGGERED",
            current_champion=_current_champion["version"],
            triggered_by="admin",
        )

        # In production, would trigger Kubernetes or load balancer to revert traffic
        _deployment_history.append(
            {
                "event": "rollback",
                "timestamp": datetime.utcnow().isoformat(),
                "from_version": _current_champion["version"],
                "to_version": "v0.9",  # Stub: previous version
            }
        )

        return {
            "status": "rollback_triggered",
            "message": f"Rolled back from {_current_champion['version']} to v0.9",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        log.error("Rollback failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Rollback failed")


@router.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """
    Get current champion model information.

    Returns version, deployment info, and recent metrics.
    """
    try:
        return {
            "champion": {
                "version": _current_champion["version"],
                "adapter_path": _current_champion["adapter_path"],
                "deployed_at": _current_champion["deployed_at"],
                "metrics": {
                    "reward_accuracy": _current_champion["reward_accuracy"],
                    "dpo_win_rate": _current_champion["dpo_win_rate"],
                },
            },
            "deployment_history": _deployment_history[-10:],  # Last 10 events
        }
    except Exception as e:
        log.error("Failed to get model info", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve model info")


@router.post("/update-champion")
async def update_champion(
    version: str,
    adapter_path: str,
    reward_accuracy: float,
    dpo_win_rate: float,
    x_admin_key: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Update the current champion model (called by orchestrator after deployment).

    Requires X-Admin-Key header.
    """
    if not validate_admin_key(x_admin_key):
        log.warning("Unauthorized update attempt")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing admin key")

    try:
        log.info(
            "Updating champion model",
            new_version=version,
            new_adapter_path=adapter_path,
        )

        _deployment_history.append(
            {
                "event": "deployment",
                "timestamp": datetime.utcnow().isoformat(),
                "old_version": _current_champion["version"],
                "new_version": version,
                "metrics": {
                    "reward_accuracy": reward_accuracy,
                    "dpo_win_rate": dpo_win_rate,
                },
            }
        )

        _current_champion.update(
            {
                "version": version,
                "adapter_path": adapter_path,
                "deployed_at": datetime.utcnow().isoformat(),
                "reward_accuracy": reward_accuracy,
                "dpo_win_rate": dpo_win_rate,
            }
        )

        return {
            "status": "champion_updated",
            "new_version": version,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        log.error("Failed to update champion", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update champion model")
