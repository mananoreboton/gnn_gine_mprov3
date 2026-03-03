"""
Train GNN on MPro Version 3: regression (pIC50) and optional classification (Category).
Requires a pre-built PyG dataset (run build_dataset.py first). Train/val/test from three split files.
Usage:
  uv run python build_dataset.py --data_root /path/to/snapshot
  python train.py --data_root /path/to/snapshot [--num_folds 5] [--fold_index 0] [--epochs 100]
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PYG_DATASET_NAME,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_VAL_SPLIT_FILE,
    DEFAULT_TEST_SPLIT_FILE,
    SplitConfig,
    TrainingConfig,
)
from gine_config import GineConfig
from loaders import create_data_loaders
from training import train_one_epoch
from validation import evaluate_validation
from testing import evaluate_test, print_test_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN on MPro Version 3")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to MPro-URV_Version3_snapshot (default: ../MPro-URV_Version3_snapshot)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_PYG_DATASET_NAME,
        help=f"PyG dataset folder name under data_root (default: {DEFAULT_PYG_DATASET_NAME})",
    )
    parser.add_argument(
        "--train_split_file",
        type=str,
        default=DEFAULT_TRAIN_SPLIT_FILE,
        help=f"Train split file in Splits/ (default: {DEFAULT_TRAIN_SPLIT_FILE})",
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default=DEFAULT_VAL_SPLIT_FILE,
        help=f"Val split file in Splits/ (default: {DEFAULT_VAL_SPLIT_FILE})",
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=DEFAULT_TEST_SPLIT_FILE,
        help=f"Test split file in Splits/ (default: {DEFAULT_TEST_SPLIT_FILE})",
    )
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--fold_index", type=int, default=0, help="Which fold to use (0 .. num_folds-1)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classification", action="store_true", help="Add classification loss (Category)")
    parser.add_argument("--no_classification", action="store_false", dest="classification")
    parser.set_defaults(classification=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root or DEFAULT_DATA_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    torch.manual_seed(args.seed)

    split_config = SplitConfig(
        train_file=args.train_split_file,
        val_file=args.val_split_file,
        test_file=args.test_split_file,
        num_folds=args.num_folds,
        fold_index=args.fold_index,
        dataset_name=args.dataset_name,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        use_classification=args.classification,
        classification_loss_weight=0.5,
    )
    gine_config = GineConfig(
        in_channels=4,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        out_regression=1,
        out_classes=3 if args.classification else None,
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        data_root, split_config, batch_size=training_config.batch_size
    )
    print(f"Dataset size (train/val/test loaders): {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gine_config.build().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    best_val_rmse = float("inf")
    for epoch in range(1, training_config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion_mse,
            criterion_ce,
            training_config.use_classification,
            training_config.classification_loss_weight,
        )
        val_metrics = evaluate_validation(
            model, val_loader, device, training_config.use_classification
        )
        if val_metrics.rmse < best_val_rmse:
            best_val_rmse = val_metrics.rmse
            torch.save(model.state_dict(), data_root / "best_gnn.pt")
        if epoch % 10 == 0 or epoch == 1:
            acc_str = f"  val_acc={val_metrics.accuracy:.4f}" if val_metrics.accuracy is not None else ""
            print(
                f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_rmse={val_metrics.rmse:.4f}{acc_str}"
            )

    model.load_state_dict(torch.load(data_root / "best_gnn.pt"))
    test_metrics = evaluate_test(
        model, test_loader, device, training_config.use_classification
    )
    print_test_report(test_metrics, training_config.use_classification)


if __name__ == "__main__":
    main()
