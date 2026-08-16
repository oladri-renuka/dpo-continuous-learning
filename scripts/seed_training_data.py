"""Generate synthetic training data for Phase 1 RunPod spike (no Kafka/S3)."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

log = structlog.get_logger(__name__)


def generate_synthetic_data() -> List[Dict[str, str]]:
    """Generate 300 synthetic training examples using templates."""

    log.info("Generating synthetic training data...")

    # Question-answer templates
    qa_pairs = [
        {
            "topic": "Python",
            "correct": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms and has a large standard library.",
            "incorrect": "Python is a snake."
        },
        {
            "topic": "Machine Learning",
            "correct": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data and identify patterns.",
            "incorrect": "Machine learning is when you teach a computer to learn things."
        },
        {
            "topic": "Neural Networks",
            "correct": "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes organized in layers, where each connection has learnable weights.",
            "incorrect": "Neural networks are networks made of neurons."
        },
        {
            "topic": "Gradient Descent",
            "correct": "Gradient descent is an optimization algorithm used to minimize a loss function by iteratively moving in the direction of steepest descent. It's fundamental to training neural networks.",
            "incorrect": "Gradient descent is when you go down a gradient."
        },
        {
            "topic": "Regularization",
            "correct": "Regularization is a technique to prevent overfitting by adding a penalty term to the loss function. Common types include L1 (Lasso) and L2 (Ridge) regularization.",
            "incorrect": "Regularization makes things regular."
        },
        {
            "topic": "Backpropagation",
            "correct": "Backpropagation is an algorithm for training neural networks by computing gradients of the loss function with respect to network weights. It enables efficient gradient descent optimization.",
            "incorrect": "Backpropagation is when you go back and propagate."
        },
        {
            "topic": "Activation Function",
            "correct": "Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Common examples include ReLU, sigmoid, and tanh.",
            "incorrect": "Activation functions activate things."
        },
        {
            "topic": "Batch Normalization",
            "correct": "Batch normalization normalizes layer inputs by standardizing activations within minibatches. It accelerates training, enables higher learning rates, and improves model generalization.",
            "incorrect": "Batch normalization normalizes batches."
        },
        {
            "topic": "Transfer Learning",
            "correct": "Transfer learning is a technique where a model trained on one task is adapted for another task. It leverages pre-trained knowledge to improve performance on new tasks with limited data.",
            "incorrect": "Transfer learning is learning how to transfer things."
        },
        {
            "topic": "Attention Mechanism",
            "correct": "Attention mechanisms allow models to dynamically focus on different parts of the input. They compute weighted combinations of values based on learned query-key-value interactions.",
            "incorrect": "Attention is when you pay attention to something."
        },
    ]

    examples = []

    # Generate 5000 examples (4500 train, 500 val)
    for i in range(5000):
        qa = qa_pairs[i % len(qa_pairs)]

        prompt = f"Q: What is {qa['topic']}?"
        preferred = f"A: {qa['correct']}"
        rejected = f"A: {qa['incorrect']}"

        examples.append({
            "prompt": prompt,
            "preferred": preferred,
            "rejected": rejected,
            "metadata": {
                "topic": qa["topic"],
                "example_id": i,
            }
        })

    log.info(f"Generated {len(examples)} synthetic examples")
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
    """Generate and save synthetic training data."""
    try:
        from src.infra import setup_logging
        setup_logging(environment="production")
    except ImportError:
        pass  # Logging not required for seed script

    examples = generate_synthetic_data()
    save_data(examples)


if __name__ == "__main__":
    main()
