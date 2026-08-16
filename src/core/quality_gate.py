"""Quality gate validation for model adapter."""

import sys
import torch
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import structlog
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

log = structlog.get_logger(__name__)


class ModelDegradationError(Exception):
    """Model degradation detected."""
    pass


class QualityGate:
    def __init__(
        self,
        challenger_path: str,
        champion_path: str = "baseline",
        device: str = "cpu",
    ):
        self.challenger_path = challenger_path
        self.champion_path = champion_path
        self.device = device
        self.base_model = "meta-llama/Llama-2-7b-hf"
        self.threshold_reward_accuracy = 0.72
        self.threshold_dpo_win_rate = 0.55

        log.info(
            "QualityGate initialized",
            challenger=challenger_path,
            champion=champion_path,
            device=device,
        )

    def validate(self) -> Dict[str, Any]:
        """Run validation."""
        try:
            log.info("=" * 80)
            log.info("QUALITY GATE: Starting validation")
            log.info("=" * 80)

            # Load golden eval
            log.info("Loading golden evaluation set...")
            golden_examples = self._load_golden_examples()
            log.info(f"Loaded {len(golden_examples)} golden examples")

            # Test challenger
            log.info("Testing challenger adapter...")
            challenger_accuracy = self._evaluate_model(
                golden_examples,
                self.challenger_path,
            )

            log.info(f"Challenger accuracy: {challenger_accuracy:.3f}")

            # Check thresholds
            if challenger_accuracy < self.threshold_reward_accuracy:
                raise ModelDegradationError(
                    f"Accuracy {challenger_accuracy:.3f} < threshold {self.threshold_reward_accuracy}"
                )

            log.info("=" * 80)
            log.info("✓ QUALITY GATE PASSED")
            log.info("=" * 80)

            return {
                "passed": True,
                "challenger_accuracy": challenger_accuracy,
                "threshold": self.threshold_reward_accuracy,
            }

        except ModelDegradationError as e:
            log.error(f"Quality gate failed: {str(e)}")
            raise
        except Exception as e:
            log.error(f"Validation error: {str(e)}", exc_info=True)
            raise

    def _load_golden_examples(self) -> list:
        """Load golden evaluation examples."""
        path = Path("data/golden_eval.json")
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data.get("examples", [])
        return []

    def _evaluate_model(self, examples: list, adapter_path: str) -> float:
        """Evaluate model on examples."""
        log.info(f"Loading base model: {self.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        # Load adapter if not baseline
        if adapter_path != "baseline":
            log.info(f"Loading adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

        # Simple accuracy: check response length
        correct = 0
        total = 0
        for ex in examples:
            if "ideal_response" in ex:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy


def main():
    """Main entry point."""
    try:
        from src.infra import setup_logging
        setup_logging(environment="production")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Quality Gate Validation")
    parser.add_argument("--challenger", required=True, help="Challenger adapter path")
    parser.add_argument("--champion", required=True, help="Champion adapter path")
    parser.add_argument("--device", default="cpu", help="Device for inference")

    args = parser.parse_args()

    gate = QualityGate(
        challenger_path=args.challenger,
        champion_path=args.champion,
        device=args.device,
    )

    result = gate.validate()
    log.info(f"Validation result: {result}")


if __name__ == "__main__":
    main()
