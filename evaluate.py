"""
Generic GNN evaluation: load a checkpoint and evaluate on test set using index file.
Usage:
  uv run python evaluate.py --data_root /path/to/data --processed_dir my_dataset \\
    --test_indices_file test.txt --checkpoint best_gnn.pt
For MPro v3, use: uv run python -m mpro.evaluate
"""

import argparse
from pathlib import Path

import torch

from dataset_base import GenericPyGDataset, load_indices_from_file
from gine_config import GineConfig
from loaders import create_data_loaders
from run_training import run_evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GNN on the test set (generic, uses index file)."
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to data root")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="processed_pyg",
        help="Subfolder under data_root with data.pt (default: processed_pyg)",
    )
    parser.add_argument(
        "--test_indices_file",
        type=str,
        required=True,
        help="Path to file with test indices (one int per line)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (relative to data_root or absolute)",
    )
    parser.add_argument(
        "--label_attr",
        type=str,
        default="y",
        help="Batch attribute for class labels (default: y)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64, help="Must match trained model")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    test_idx_path = Path(args.test_indices_file)
    if not test_idx_path.exists():
        raise FileNotFoundError(f"Test indices file not found: {test_idx_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = data_root / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_indices = load_indices_from_file(test_idx_path)
    dataset = GenericPyGDataset(root=str(data_root), processed_dir=args.processed_dir)
    _, _, test_loader = create_data_loaders(
        dataset, [], [], test_indices, batch_size=args.batch_size
    )
    print(f"Test set size: {len(test_indices)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gine_config = GineConfig(
        in_channels=args.in_channels,
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
        label_attr=args.label_attr,
    )


if __name__ == "__main__":
    main()
