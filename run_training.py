"""
Shared training and evaluation runners: build model, run epochs, save best; load checkpoint and evaluate.
Used by both generic (train.py, evaluate.py) and dataset-specific (e.g. mpro) scripts.
"""

from pathlib import Path

import torch
import torch.nn as nn

from gine_config import GineConfig
from model import MProGNN
from train_epoch import train_one_epoch
from validation import evaluate_validation
from evaluation import evaluate_test, print_test_report


def run_training(
    train_loader,
    val_loader,
    gine_config: GineConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    save_path: Path,
    label_attr: str = "y",
) -> None:
    """
    Run full training loop and save best model by validation accuracy.
    loaders are PyG DataLoaders; each batch has .x, .edge_index, .batch, and the label at batch.<label_attr>.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gine_config.build().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, criterion_ce, label_attr=label_attr
        )
        val_metrics = evaluate_validation(model, val_loader, device, label_attr=label_attr)
        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            torch.save(model.state_dict(), save_path)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_acc={val_metrics.accuracy:.4f}"
            )


def run_evaluation(
    test_loader,
    checkpoint_path: Path,
    gine_config: GineConfig,
    device: torch.device,
    label_attr: str = "y",
) -> None:
    """Load model from checkpoint, evaluate on test_loader, print report."""
    model = gine_config.build().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    test_metrics = evaluate_test(model, test_loader, device, label_attr=label_attr)
    print_test_report(test_metrics)
