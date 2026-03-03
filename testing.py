"""
Testing logic: run final evaluation on the test set and report metrics.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from model import MProGNN
from validation import evaluate_validation


@dataclass(frozen=True)
class TestMetrics:
    """Test set metrics (RMSE for pIC50, optional classification accuracy)."""

    rmse: float
    accuracy: Optional[float] = None


def evaluate_test(
    model: MProGNN,
    loader: DataLoader,
    device: torch.device,
    use_classification: bool = False,
) -> TestMetrics:
    """Compute test RMSE and optionally classification accuracy."""
    metrics = evaluate_validation(model, loader, device, use_classification)
    return TestMetrics(rmse=metrics.rmse, accuracy=metrics.accuracy)


def print_test_report(metrics: TestMetrics, use_classification: bool = False) -> None:
    """Print test RMSE and optional accuracy to stdout."""
    print(f"Test RMSE (pIC50): {metrics.rmse:.4f}")
    if use_classification and metrics.accuracy is not None:
        print(f"Test accuracy (Category): {metrics.accuracy:.4f}")
