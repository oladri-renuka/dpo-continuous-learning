"""Blue/Green deployment orchestration with canary rollout."""

import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import structlog

from src.infra import get_logger
from src.models.config import get_settings

log = get_logger(__name__)

# Champion pointer file (simulates model registry)
CHAMPION_POINTER_FILE = Path("/tmp/dpo-champion-pointer.json")


class BlueGreenDeployer:
    """
    Orchestrates Blue/Green deployment with canary rollout.

    - BLUE = current champion (receives 100% traffic)
    - GREEN = challenger (gradually receives traffic)
    """

    def __init__(self):
        """Initialize deployer."""
        self.settings = get_settings()
        self.current_champion_version = self._load_champion_pointer()

    def _load_champion_pointer(self) -> str:
        """Load current champion version from pointer file."""
        try:
            if CHAMPION_POINTER_FILE.exists():
                with open(CHAMPION_POINTER_FILE, "r") as f:
                    data = json.load(f)
                    return data.get("version", "v1.0")
        except Exception as e:
            log.warning("Failed to load champion pointer", error=str(e))
        return "v1.0"

    def _save_champion_pointer(self, version: str, adapter_path: str) -> None:
        """Save champion version and adapter path to pointer file."""
        try:
            CHAMPION_POINTER_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CHAMPION_POINTER_FILE, "w") as f:
                json.dump(
                    {
                        "version": version,
                        "adapter_path": adapter_path,
                        "updated_at": datetime.utcnow().isoformat(),
                    },
                    f,
                )
            log.info("Updated champion pointer", version=version)
        except Exception as e:
            log.error("Failed to save champion pointer", error=str(e))

    def _health_check(self, pod_name: str, port: int = 8000) -> bool:
        """
        Check if a pod is healthy.

        Simulated: just returns True with some randomness for testing.
        """
        log.debug(f"Health check: {pod_name}:8000/health")

        # Simulate health check (95% success rate)
        is_healthy = random.random() < 0.95
        log.info(
            f"Health check {pod_name}",
            healthy=is_healthy,
        )
        return is_healthy

    def _simulate_traffic_shift(
        self,
        green_traffic_pct: int,
        green_success_rate: float,
    ) -> bool:
        """
        Simulate traffic shift and monitor metrics.

        Returns True if metrics are healthy, False if degradation detected.
        """
        log.info(
            "Canary monitoring",
            green_traffic_pct=green_traffic_pct,
            baseline_success_rate=0.995,
            green_success_rate=f"{green_success_rate:.3f}",
        )

        # Check if metrics are acceptable
        if green_success_rate < 0.95:
            log.error(
                "Canary degradation detected",
                green_success_rate=green_success_rate,
                threshold=0.95,
            )
            return False

        return True

    def deploy_canary(
        self,
        challenger_adapter_path: str,
        challenger_version: str = "v1.1",
        timeout_seconds: int = 300,
    ) -> Dict[str, Any]:
        """
        Deploy challenger model via canary rollout.

        Timeline:
        - 0-60s: 10% → GREEN, 90% → BLUE (initial canary)
        - 60-120s: 50% → GREEN, 50% → BLUE (ramp up)
        - 120-300s: 100% → GREEN, 0% → BLUE (full rollout)

        Returns: deployment result with status and metrics.
        """
        log.info(
            "=" * 80,
        )
        log.info("BLUE/GREEN DEPLOYMENT: Starting canary rollout")
        log.info("=" * 80)

        deployment_start = time.time()
        blue_version = self.current_champion_version
        green_version = challenger_version

        try:
            # === Phase 1: Spin up GREEN pod ===
            log.info("PHASE 1: Spinning up GREEN pod", version=green_version)
            log.info(f"GREEN: Starting pod with adapter {challenger_adapter_path}")

            # Health check
            time.sleep(1)  # Simulate startup time
            if not self._health_check("GREEN", 8001):
                log.error("GREEN pod failed health check, aborting deployment")
                return {
                    "status": "aborted",
                    "reason": "GREEN health check failed",
                    "duration_seconds": time.time() - deployment_start,
                }

            log.info("✓ GREEN pod healthy")

            # === Phase 2: Canary 10% traffic (0-60s) ===
            log.info("PHASE 2: Canary - 10% traffic → GREEN, 90% → BLUE")
            canary_start = time.time()
            green_success_rate = 0.99  # Simulate 99% success rate

            while time.time() - canary_start < 60:
                if not self._simulate_traffic_shift(10, green_success_rate):
                    log.error("Canary degradation at 10%, initiating rollback")
                    return {
                        "status": "rollback",
                        "reason": "Canary metrics degradation at 10% traffic",
                        "duration_seconds": time.time() - deployment_start,
                    }
                time.sleep(15)

            log.info("✓ Canary phase 1 passed")

            # === Phase 3: Ramp up 50% traffic (60-120s) ===
            log.info("PHASE 3: Ramping up - 50% traffic → GREEN, 50% → BLUE")
            ramp_start = time.time()

            while time.time() - ramp_start < 60:
                if not self._simulate_traffic_shift(50, green_success_rate):
                    log.error("Canary degradation at 50%, initiating rollback")
                    return {
                        "status": "rollback",
                        "reason": "Canary metrics degradation at 50% traffic",
                        "duration_seconds": time.time() - deployment_start,
                    }
                time.sleep(15)

            log.info("✓ Canary phase 2 passed")

            # === Phase 4: Full rollout 100% traffic (120-300s) ===
            log.info("PHASE 4: Full rollout - 100% traffic → GREEN, 0% → BLUE")
            rollout_start = time.time()

            while time.time() - rollout_start < 180:
                if not self._simulate_traffic_shift(100, green_success_rate):
                    log.error("Degradation detected during full rollout, initiating rollback")
                    return {
                        "status": "rollback",
                        "reason": "Metrics degradation during full rollout",
                        "duration_seconds": time.time() - deployment_start,
                    }
                time.sleep(30)

            # === Deployment Successful ===
            log.info("=" * 80)
            log.info("✓✓✓ DEPLOYMENT SUCCESSFUL ✓✓✓")
            log.info("=" * 80)

            # Update champion pointer
            self._save_champion_pointer(green_version, challenger_adapter_path)
            self.current_champion_version = green_version

            total_duration = time.time() - deployment_start

            return {
                "status": "success",
                "champion_version": green_version,
                "challenger_version": green_version,
                "previous_champion": blue_version,
                "duration_seconds": round(total_duration, 2),
                "adapter_path": challenger_adapter_path,
            }

        except Exception as e:
            log.error("Deployment failed", error=str(e), exc_info=True)
            return {
                "status": "failed",
                "reason": str(e),
                "duration_seconds": time.time() - deployment_start,
            }


def deploy_model(
    challenger_adapter_path: str,
    challenger_version: str = "v1.1",
) -> Dict[str, Any]:
    """
    Public function to trigger deployment.

    Called by orchestrator after quality gate passes.
    """
    deployer = BlueGreenDeployer()
    return deployer.deploy_canary(
        challenger_adapter_path=challenger_adapter_path,
        challenger_version=challenger_version,
    )
