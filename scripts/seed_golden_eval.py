"""Seed golden evaluation set to S3 (production only)."""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.schemas import GoldenEvalExample, GoldenEvalSet
from src.infra.s3_client import S3Client




def seed_golden_eval(bucket: str = "ml-artifacts", version: str = "v1") -> None:
    """
    Generate and upload golden evaluation set to S3 (production only).

    Args:
        bucket: S3 bucket name
        version: Version identifier (v1, v2, etc.)

    Raises:
        S3ClientError: If S3 upload fails
    """
    s3_client = S3Client(bucket=bucket)

    # Create golden eval examples
    examples = [
        # Easy examples (clear preference signal)
        GoldenEvalExample(
            id="golden_001",
            prompt="What is machine learning?",
            preferred="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention.",
            rejected="Machine learning is about computers learning stuff.",
            category="education",
            difficulty="easy",
            human_rater="expert_001",
            expected_score=0.95,
            notes="Clear preference for comprehensive definition",
        ),
        GoldenEvalExample(
            id="golden_002",
            prompt="Explain supervised learning with an example.",
            preferred="Supervised learning is a machine learning paradigm where the model is trained on labeled data. For example, in email spam classification, the system learns from thousands of emails labeled as either 'spam' or 'not spam' to build a classifier that can predict labels for new, unseen emails.",
            rejected="Supervised learning uses labeled data. Example: spam emails.",
            category="education",
            difficulty="easy",
            human_rater="expert_001",
            expected_score=0.92,
        ),
        GoldenEvalExample(
            id="golden_003",
            prompt="What are neural networks?",
            preferred="Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers. Each connection has a learnable weight, and neurons apply activation functions to produce outputs. This architecture enables deep learning models to learn complex, non-linear relationships in data.",
            rejected="Neural networks are networks made of neurons.",
            category="technical",
            difficulty="easy",
            human_rater="expert_002",
            expected_score=0.94,
        ),
        # Hard examples (borderline preferences)
        GoldenEvalExample(
            id="golden_004",
            prompt="Compare batch normalization and layer normalization.",
            preferred="Batch normalization normalizes inputs across the batch dimension, reducing internal covariate shift during training. Layer normalization normalizes across features for each sample independently. BN improves training speed but has different train/test behavior. LN is more stable for RNNs and doesn't depend on batch size.",
            rejected="Batch normalization normalizes batches. Layer normalization normalizes layers. They're different.",
            category="technical",
            difficulty="hard",
            human_rater="expert_002",
            expected_score=0.78,
            notes="Subtle but important technical distinctions",
        ),
        GoldenEvalExample(
            id="golden_005",
            prompt="Discuss the trade-offs of regularization techniques.",
            preferred="L1 regularization encourages sparsity (useful for feature selection) but is non-differentiable at zero. L2 regularization is differentiable and reduces weight magnitudes uniformly. Dropout is effective for preventing co-adaptation but adds computational overhead during training. The choice depends on the problem structure and interpretability requirements.",
            rejected="L1 and L2 are regularization. Dropout is also regularization. They prevent overfitting.",
            category="technical",
            difficulty="hard",
            human_rater="expert_001",
            expected_score=0.76,
        ),
        GoldenEvalExample(
            id="golden_006",
            prompt="What is reinforcement learning used for?",
            preferred="Reinforcement learning trains agents to maximize cumulative rewards through interaction with an environment. Applications include game playing (AlphaGo), robotics control, autonomous driving, and optimization problems. The agent learns through trial-and-error, balancing exploration of new actions with exploitation of known good actions.",
            rejected="Reinforcement learning is about learning from rewards.",
            category="education",
            difficulty="easy",
            human_rater="expert_003",
            expected_score=0.91,
        ),
        GoldenEvalExample(
            id="golden_007",
            prompt="Explain attention mechanisms in transformers.",
            preferred="Attention mechanisms compute a weighted sum of value vectors based on the similarity between query and key vectors. In transformers, multi-head attention allows the model to attend to different representation subspaces. Self-attention enables parallel processing and captures long-range dependencies. The softmax normalizes attention weights to create a probability distribution.",
            rejected="Attention is about paying attention to important parts.",
            category="technical",
            difficulty="hard",
            human_rater="expert_002",
            expected_score=0.85,
        ),
        GoldenEvalExample(
            id="golden_008",
            prompt="How does gradient descent work?",
            preferred="Gradient descent is an optimization algorithm that iteratively updates parameters in the direction of steepest descent (negative gradient) to minimize a loss function. The learning rate controls step size. Variants include batch GD (all data), SGD (one sample), and mini-batch GD (subset). Momentum and adaptive methods (Adam, RMSprop) improve convergence.",
            rejected="Gradient descent uses gradients to go downhill.",
            category="education",
            difficulty="easy",
            human_rater="expert_001",
            expected_score=0.93,
        ),
        GoldenEvalExample(
            id="golden_009",
            prompt="Discuss overfitting and methods to prevent it.",
            preferred="Overfitting occurs when a model learns noise in training data and fails to generalize. Prevention methods include: regularization (L1/L2), dropout, early stopping, data augmentation, cross-validation, and using simpler models. The bias-variance tradeoff is central—increased model capacity increases variance while reducing bias.",
            rejected="Overfitting is when the model memorizes data. Use dropout or regularization.",
            category="education",
            difficulty="medium",
            human_rater="expert_003",
            expected_score=0.88,
        ),
        GoldenEvalExample(
            id="golden_010",
            prompt="What is the purpose of backpropagation?",
            preferred="Backpropagation computes gradients of the loss function with respect to model parameters by applying the chain rule through the network layers in reverse order. These gradients enable gradient-based optimization algorithms (like SGD) to update weights. Efficient backpropagation (via automatic differentiation) is foundational to training deep neural networks.",
            rejected="Backpropagation propagates errors backward through the network.",
            category="technical",
            difficulty="medium",
            human_rater="expert_002",
            expected_score=0.89,
        ),
    ]

    # Create GoldenEvalSet
    golden_eval_set = GoldenEvalSet(
        version=version,
        examples=examples,
        total_examples=len(examples),
        metadata={
            "created_by": "seed_golden_eval.py",
            "creation_date": datetime.now(timezone.utc).isoformat(),
            "description": "Mock golden evaluation set for local testing",
            "easy_examples": sum(1 for ex in examples if ex.difficulty == "easy"),
            "medium_examples": sum(1 for ex in examples if ex.difficulty == "medium"),
            "hard_examples": sum(1 for ex in examples if ex.difficulty == "hard"),
        },
    )

    # Upload examples to S3
    examples_key = f"golden_eval/{version}/examples.json"
    examples_uri = s3_client.upload_json(
        examples_key,
        {"examples": [ex.model_dump() for ex in examples]},
    )
    print(f"✓ Uploaded {len(examples)} examples to {examples_uri}")

    # Upload metadata to S3
    metadata_key = f"golden_eval/{version}/metadata.json"
    metadata_uri = s3_client.upload_json(metadata_key, golden_eval_set.metadata)
    print(f"✓ Uploaded metadata to {metadata_uri}")

    print(f"\nGolden eval set seeded successfully to S3:")
    print(f"  Version: {version}")
    print(f"  Total examples: {len(examples)}")
    print(f"  S3 bucket: {bucket}")


if __name__ == "__main__":
    seed_golden_eval()
