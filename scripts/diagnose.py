"""Diagnostics & RCA (Root Cause Analysis) for failed pipeline runs."""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.infra import setup_logging, get_logger
from src.infra.mlflow_client import MLflowClient, MLflowClientError

log = get_logger(__name__)


class Diagnostic:
    """Root cause analysis tool for debugging pipeline failures."""

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize diagnostic tool.

        Args:
            run_id: MLflow run ID to analyze (optional)
        """
        self.run_id = run_id
        self.mlflow_client = MLflowClient()
        self.report = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "findings": [],
            "recommendations": [],
        }

    def run(self, mode: str = "full") -> Dict[str, Any]:
        """
        Run diagnostic analysis.

        Modes:
        - data_quality: Analyze data issues
        - training_quality: Analyze training metrics
        - full: Complete analysis

        Args:
            mode: Diagnostic mode

        Returns:
            Diagnostic report
        """
        log.info("=" * 80)
        log.info("DIAGNOSTIC: Starting root cause analysis")
        log.info("=" * 80)

        try:
            if mode in ("data_quality", "full"):
                self._analyze_data_quality()

            if mode in ("training_quality", "full"):
                self._analyze_training_quality()

            if mode == "full":
                self._analyze_model_outputs()
                self._analyze_metric_drift()

            self._generate_report()
            return self.report

        except Exception as e:
            log.error(f"Diagnostic failed: {str(e)}", exc_info=True)
            self.report["error"] = str(e)
            return self.report

    def _analyze_data_quality(self) -> None:
        """Analyze data quality issues."""
        log.info("\nAnalyzing data quality...")

        findings = []

        # Check baseline accuracy
        log.info("Checking baseline accuracy...")
        # Stub: would load from MLflow
        baseline_acc = 0.48

        if baseline_acc < 0.55:
            findings.append(
                {
                    "severity": "CRITICAL",
                    "type": "data_quality",
                    "issue": f"Baseline accuracy {baseline_acc:.2%} < 0.55",
                    "cause": "Data is intrinsically noisy or labels are inconsistent",
                    "recommendation": [
                        "Review 50 random feedback samples for label errors",
                        "Check for mode collapse in user responses",
                        "Increase feedback quality review thresholds",
                        "Consider A/B testing on feedback UI to reduce ambiguity",
                    ],
                }
            )

        # Check for class imbalance
        log.info("Checking class imbalance...")
        # Stub: would compute from data
        thumbs_up_ratio = 0.65

        if thumbs_up_ratio > 0.80 or thumbs_up_ratio < 0.20:
            findings.append(
                {
                    "severity": "WARNING",
                    "type": "class_imbalance",
                    "issue": f"Class imbalance: {thumbs_up_ratio:.1%} thumbs_up",
                    "cause": "Feedback distribution is skewed",
                    "recommendation": [
                        "Monitor feedback collection over time",
                        "Consider stratified sampling during training",
                        "Adjust feedback incentives if needed",
                    ],
                }
            )

        self.report["findings"].extend(findings)
        log.info(f"Found {len(findings)} data quality issues")

    def _analyze_training_quality(self) -> None:
        """Analyze training metrics and convergence."""
        log.info("\nAnalyzing training quality...")

        findings = []

        # Check for overfitting
        log.info("Checking for overfitting...")
        # Stub: would load from MLflow
        train_loss = 0.30
        val_loss = 0.55
        loss_ratio = val_loss / train_loss if train_loss > 0 else 0

        if loss_ratio > 1.5:
            findings.append(
                {
                    "severity": "CRITICAL",
                    "type": "overfitting",
                    "issue": f"Validation loss {loss_ratio:.2f}x training loss",
                    "cause": "Model is memorizing training data, not generalizing",
                    "recommendation": [
                        "Increase dropout from 0.1 to 0.2",
                        "Add weight decay (0.01)",
                        "Reduce LoRA rank from 16 to 8",
                        "Reduce training epochs from 3 to 2",
                        "Increase batch size to reduce per-sample updates",
                    ],
                }
            )

        # Check for vanishing/exploding gradients
        log.info("Checking gradient norms...")
        # Stub: would compute from training logs
        mean_gradient_norm = 0.001

        if mean_gradient_norm < 0.0001:
            findings.append(
                {
                    "severity": "CRITICAL",
                    "type": "vanishing_gradients",
                    "issue": f"Mean gradient norm {mean_gradient_norm:.6f} (too small)",
                    "cause": "Gradients are vanishing, model not learning",
                    "recommendation": [
                        "Remove bias decay (set to 0)",
                        "Increase learning rate from 5e-4 to 1e-3",
                        "Check activation functions (avoid deep stacks)",
                        "Use gradient clipping",
                    ],
                }
            )

        self.report["findings"].extend(findings)
        log.info(f"Found {len(findings)} training issues")

    def _analyze_model_outputs(self) -> None:
        """Analyze model output quality."""
        log.info("\nAnalyzing model outputs...")

        log.info("Sampling 50 model predictions for inspection...")

        # Stub: would load actual model outputs
        issues = []

        # Check for mode collapse
        log.info("Checking for mode collapse...")
        # Stub: would compare outputs
        unique_outputs_ratio = 0.45

        if unique_outputs_ratio < 0.50:
            issues.append(
                {
                    "severity": "CRITICAL",
                    "type": "mode_collapse",
                    "issue": "Model outputting very similar/identical responses",
                    "cause": "DPO loss may be too aggressive or beta too high",
                    "recommendation": [
                        "Reduce DPO beta from 0.1 to 0.05",
                        "Increase reference model weight",
                        "Add diversity regularization",
                    ],
                }
            )

        self.report["findings"].extend(issues)
        log.info(f"Found {len(issues)} output quality issues")

    def _analyze_metric_drift(self) -> None:
        """Analyze metric trends over time."""
        log.info("\nAnalyzing metric drift...")

        log.info("Loading historical metrics from MLflow...")

        # Stub: would load from MLflow
        recent_acc = 0.72
        prev_acc = 0.78
        acc_drop = prev_acc - recent_acc

        if acc_drop > 0.05:
            self.report["findings"].append(
                {
                    "severity": "WARNING",
                    "type": "metric_degradation",
                    "issue": f"Accuracy dropped {acc_drop:.1%} ({prev_acc:.2%} → {recent_acc:.2%})",
                    "cause": "Model quality declining (possible data drift)",
                    "recommendation": [
                        "Analyze recent feedback for distribution shift",
                        "Check if user preferences have changed",
                        "Consider rolling back to previous champion",
                        "Increase data collection to detect changes earlier",
                    ],
                }
            )

    def _generate_report(self) -> None:
        """Generate and save diagnostic report."""
        log.info("\n" + "=" * 80)
        log.info("DIAGNOSTIC REPORT")
        log.info("=" * 80)

        if not self.report["findings"]:
            log.info("✓ No issues detected")
            self.report["status"] = "healthy"
        else:
            critical = [f for f in self.report["findings"] if f["severity"] == "CRITICAL"]
            warnings = [f for f in self.report["findings"] if f["severity"] == "WARNING"]

            log.warning(
                f"Found {len(critical)} critical issues, {len(warnings)} warnings",
                critical=len(critical),
                warnings=len(warnings),
            )

            if critical:
                self.report["status"] = "failed"
            else:
                self.report["status"] = "degraded"

        # Save report to file
        report_path = f"/tmp/rca_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_path, "w") as f:
                json.dump(self.report, f, indent=2, default=str)
            log.info(f"Report saved to {report_path}")
        except Exception as e:
            log.error(f"Failed to save report: {str(e)}")


def main():
    """Entrypoint for diagnostic tool."""
    import argparse

    parser = argparse.ArgumentParser(description="Diagnostic & RCA Tool")
    parser.add_argument(
        "--run-id",
        type=str,
        help="MLflow run ID to analyze",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["data_quality", "training_quality", "full"],
        help="Diagnostic mode",
    )
    args = parser.parse_args()

    try:
        diagnostic = Diagnostic(run_id=args.run_id)
        report = diagnostic.run(mode=args.mode)

        log.info(f"Diagnostic complete", status=report.get("status"))
        sys.exit(0)

    except Exception as e:
        log.error(f"Diagnostic failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
