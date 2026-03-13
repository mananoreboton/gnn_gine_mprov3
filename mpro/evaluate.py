"""
Evaluate a trained GNN on the MPro v3 test set.
Usage: uv run python -m mpro.evaluate [--data_root /path] [--checkpoint best_gnn.pt] [--fold_index 0]
"""

import argparse
from pathlib import Path

import torch

from gine_config import GineConfig
from loaders import create_data_loaders
from mpro.config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PYG_DATASET_NAME,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_VAL_SPLIT_FILE,
    DEFAULT_TEST_SPLIT_FILE,
    SplitConfig,
)
from mpro.dataset import MProV3Dataset, get_train_val_test_indices, load_dataset_pdb_order
from run_training import run_evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained GNN on MPro v3 test set")
    parser.add_argument("--data_root", type=str, default=None, help=f"Default: {DEFAULT_DATA_ROOT}")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_PYG_DATASET_NAME)
    parser.add_argument("--train_split_file", type=str, default=DEFAULT_TRAIN_SPLIT_FILE)
    parser.add_argument("--val_split_file", type=str, default=DEFAULT_VAL_SPLIT_FILE)
    parser.add_argument("--test_split_file", type=str, default=DEFAULT_TEST_SPLIT_FILE)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="best_gnn.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_classes", type=int, default=3)
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
    dataset = MProV3Dataset(root=str(data_root), dataset_name=split_config.dataset_name)
    dataset_pdb_order = load_dataset_pdb_order(data_root, split_config.dataset_name)
    train_idx, val_idx, test_idx = get_train_val_test_indices(
        data_root,
        split_config.train_file,
        split_config.val_file,
        split_config.test_file,
        split_config.num_folds,
        split_config.fold_index,
        dataset_pdb_order=dataset_pdb_order,
    )
    _, _, test_loader = create_data_loaders(
        dataset,
        train_idx.tolist(),
        val_idx.tolist(),
        test_idx.tolist(),
        batch_size=args.batch_size,
    )
    print(f"Test set size: {len(test_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gine_config = GineConfig(
        in_channels=4,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        out_classes=args.num_classes,
    )
    run_evaluation(
        test_loader,
        checkpoint_path,
        gine_config,
        device,
        label_attr="category",
    )


if __name__ == "__main__":
    main()
