"""
Evaluate a trained GNN on the test set from the command line.
Loads a saved checkpoint and reports test RMSE and optional classification accuracy.
Use the same split/fold and model architecture as training.
Usage:
  uv run python evaluate.py --data_root /path/to/snapshot [--checkpoint best_gnn.pt] [--fold_index 0]
"""

import argparse
from pathlib import Path

import torch

from config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PYG_DATASET_NAME,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_VAL_SPLIT_FILE,
    DEFAULT_TEST_SPLIT_FILE,
    SplitConfig,
)
from gine_config import GineConfig
from loaders import create_data_loaders
from evaluation import evaluate_test, print_test_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GNN on the test set (run independently of training)."
    )
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
        "--checkpoint",
        type=str,
        default="best_gnn.pt",
        help="Path to model checkpoint relative to data_root, or absolute path (default: best_gnn.pt)",
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--hidden", type=int, default=64, help="Must match trained model")
    parser.add_argument("--num_layers", type=int, default=3, help="Must match trained model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Must match trained model")
    parser.add_argument("--classification", action="store_true", help="Model has classification head")
    parser.add_argument("--no_classification", action="store_false", dest="classification")
    parser.set_defaults(classification=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root or DEFAULT_DATA_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = data_root / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    split_config = SplitConfig(
        train_file=args.train_split_file,
        val_file=args.val_split_file,
        test_file=args.test_split_file,
        num_folds=args.num_folds,
        fold_index=args.fold_index,
        dataset_name=args.dataset_name,
    )
    gine_config = GineConfig(
        in_channels=4,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        out_regression=1,
        out_classes=3 if args.classification else None,
    )

    _, _, test_loader = create_data_loaders(
        data_root, split_config, batch_size=args.batch_size
    )
    print(f"Test set size: {len(test_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gine_config.build().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))

    test_metrics = evaluate_test(
        model, test_loader, device, use_classification=args.classification
    )
    print_test_report(test_metrics, args.classification)


if __name__ == "__main__":
    main()
