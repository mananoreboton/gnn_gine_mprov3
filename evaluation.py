"""
Evaluation logic: run model on a dataset (e.g. test set) and compute classification metrics.
Provides evaluate_test(), TestMetrics, and print_test_report. Used by evaluate.py (evaluation CLI).
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from model import MProGNN
from validation import evaluate_validation


@dataclass(frozen=True)
class TestMetrics:
    """Test set metrics (classification accuracy)."""

    accuracy: float


def evaluate_test(
    model: MProGNN,
    loader: DataLoader,
    device: torch.device,
) -> TestMetrics:
    """Compute test accuracy."""
    metrics = evaluate_validation(model, loader, device)
    return TestMetrics(accuracy=metrics.accuracy)


def print_test_report(metrics: TestMetrics) -> None:
    """Print test accuracy to stdout."""
    print(f"Test accuracy (Category): {metrics.accuracy:.4f}")
