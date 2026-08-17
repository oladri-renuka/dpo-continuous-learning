"""Load real Anthropic/hh-rlhf dataset for training."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import structlog
from datasets import load_dataset

log = structlog.get_logger(__name__)


def load_real_dataset() -> List[Dict[str, str]]:
    """Load real Anthropic/hh-rlhf dataset from HuggingFace."""

    log.info("Loading Anthropic/hh-rlhf dataset from HuggingFace...")

    try:
        # Load real dataset
        dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=False)
        log.info(f"Loaded {len(dataset)} examples from Anthropic/hh-rlhf")

        examples = []
        for i, item in enumerate(dataset):
            if i >= 5000:  # Limit to 5000 for Phase 1
                break

            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")

            if not chosen or not rejected:
                continue

            # hh-rlhf format: full conversation in chosen/rejected
            # Format: "\n\nHuman: <question>\n\nAssistant: <response>"
            # Extract prompt (everything up to last "Assistant:")
            import re

            # Find the last occurrence of "\n\nAssistant:"
            chosen_parts = chosen.split("\n\nAssistant:")
            rejected_parts = rejected.split("\n\nAssistant:")

            if len(chosen_parts) >= 2 and len(rejected_parts) >= 2:
                # Reconstruct prompt from chosen (they should have same prompt)
                prompt = "\n\nAssistant:".join(chosen_parts[:-1]) + "\n\nAssistant:"
                chosen_response = chosen_parts[-1].strip()
                rejected_response = rejected_parts[-1].strip()

                if prompt and chosen_response and rejected_response:
                    examples.append(
                        {
                            "prompt": prompt,
                            "preferred": chosen_response,
                            "rejected": rejected_response,
                            "metadata": {
                                "source": "anthropic/hh-rlhf",
                                "example_id": i,
                            },
                        }
                    )

        log.info(f"Extracted {len(examples)} valid examples from dataset")
        return examples

    except Exception as e:
        log.error(f"Failed to load real dataset: {str(e)}")
        log.info("Falling back to synthetic data...")
        return load_synthetic_data()


def load_synthetic_data() -> List[Dict[str, str]]:
    """Fallback: Generate synthetic data if real dataset unavailable."""

    log.warning("Using synthetic data as fallback")

    qa_pairs = [
        {
            "topic": "Python",
            "correct": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms and has a large standard library.",
            "incorrect": "Python is a snake."
        },
        {
            "topic": "Machine Learning",
            "correct": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "incorrect": "Machine learning is teaching computers to do things."
        },
    ]

    examples = []
    for i in range(5000):
        qa = qa_pairs[i % len(qa_pairs)]

        examples.append({
            "prompt": f"Q: What is {qa['topic']}?",
            "preferred": f"A: {qa['correct']}",
            "rejected": f"A: {qa['incorrect']}",
            "metadata": {
                "topic": qa["topic"],
                "example_id": i,
                "source": "synthetic",
            }
        })

    return examples


def save_data(examples: List[Dict[str, str]], output_dir: str = "./data") -> None:
    """Save training and validation data to JSONL files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split: 4500 train, 500 val
    train_examples = examples[:4500]
    val_examples = examples[4500:]

    # Save train.jsonl
    train_path = output_path / "train.jsonl"
    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    log.info(f"Saved {len(train_examples)} training examples to {train_path}")

    # Save val.jsonl
    val_path = output_path / "val.jsonl"
    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")
    log.info(f"Saved {len(val_examples)} validation examples to {val_path}")

    print(f"\n✓ Data seeded successfully:")
    print(f"  Train: {train_path} ({len(train_examples)} examples)")
    print(f"  Val: {val_path} ({len(val_examples)} examples)")
    print(f"  Total: {len(examples)} examples\n")


def main():
    """Generate and save training data."""
    try:
        from src.infra import setup_logging
        setup_logging(environment="production")
    except ImportError:
        pass

    # Load real dataset (or fallback to synthetic)
    examples = load_real_dataset()

    # Save to files
    save_data(examples)


if __name__ == "__main__":
    main()
