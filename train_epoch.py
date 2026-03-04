"""
One-epoch training step: run a single training epoch and return mean loss.
Used by train.py (training CLI).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import MProGNN


def train_one_epoch(
    model: MProGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion_mse: nn.Module,
    criterion_ce: nn.Module,
    use_classification: bool,
    classification_loss_weight: float = 0.5,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        pIC50 = batch.pIC50.to(device).squeeze(-1)
        category = batch.category.to(device).squeeze(-1)
        data_batch = batch
        optimizer.zero_grad()
        edge_attr = getattr(batch, "edge_attr", None)
        pred_pIC50, logits = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr,
        )
        loss = criterion_mse(pred_pIC50.squeeze(-1), pIC50.squeeze(-1))
        if use_classification and logits is not None:
            loss = loss + classification_loss_weight * criterion_ce(logits, category)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) if len(loader) else 0.0
