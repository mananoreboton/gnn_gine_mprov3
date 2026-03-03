"""
Validation logic: evaluate the model on the validation set and compute metrics.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from model import MProGNN


@dataclass(frozen=True)
class ValidationMetrics:
    """Validation set metrics (RMSE for pIC50, optional classification accuracy)."""

    rmse: float
    accuracy: Optional[float] = None


def evaluate_validation(
    model: MProGNN,
    loader: DataLoader,
    device: torch.device,
    use_classification: bool = False,
) -> ValidationMetrics:
    """Compute validation RMSE and optionally classification accuracy."""
    model.eval()
    mse_sum = 0.0
    n = 0
    correct_cls = 0
    total_cls = 0
    with torch.no_grad():
        for data_batch, pIC50, category in loader:
            data_batch = data_batch.to(device)
            pIC50 = pIC50.to(device)
            category = category.to(device).squeeze(-1)
            edge_attr = getattr(data_batch, "edge_attr", None)
            pred_pIC50, logits = model(
                data_batch.x,
                data_batch.edge_index,
                data_batch.batch,
                edge_attr,
            )
            mse_sum += ((pred_pIC50.squeeze(-1) - pIC50.squeeze(-1)) ** 2).sum().item()
            n += pred_pIC50.size(0)
            if use_classification and logits is not None:
                pred_cls = logits.argmax(dim=1)
                correct_cls += (pred_cls == category).sum().item()
                total_cls += category.size(0)
    rmse = (mse_sum / n) ** 0.5 if n else 0.0
    acc = correct_cls / total_cls if total_cls else None
    return ValidationMetrics(rmse=rmse, accuracy=acc)
