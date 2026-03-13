"""
Generic GNN training: works with any PyG dataset and train/val/test index files.
Requires a pre-built dataset (data.pt) and three files with one integer index per line.
Usage:
  uv run python train.py --data_root /path/to/data --processed_dir my_dataset \\
    --train_indices_file train.txt --val_indices_file val.txt --test_indices_file test.txt
For MPro v3, use: uv run python -m mpro.train
"""

import argparse
from pathlib import Path

from config import TrainingConfig
from dataset_base import GenericPyGDataset, load_indices_from_file
from gine_config import GineConfig
from loaders import create_data_loaders
from run_training import run_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GNN on any PyG dataset (classification). Uses index files for splits."
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to data root (contains processed_dir)")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="processed_pyg",
        help="Subfolder under data_root with data.pt (default: processed_pyg)",
    )
    parser.add_argument(
        "--train_indices_file",
        type=str,
        required=True,
        help="Path to file with train indices (one int per line)",
    )
    parser.add_argument(
        "--val_indices_file",
        type=str,
        required=True,
        help="Path to file with validation indices (one int per line)",
    )
    parser.add_argument(
        "--test_indices_file",
        type=str,
        required=True,
        help="Path to file with test indices (one int per line)",
    )
    parser.add_argument(
        "--label_attr",
        type=str,
        default="y",
        help="Batch attribute for class labels (default: y)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in_channels", type=int, default=4, help="Node feature dimension")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Where to save best model (default: data_root/best_gnn.pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    train_idx_path = Path(args.train_indices_file)
    val_idx_path = Path(args.val_indices_file)
    test_idx_path = Path(args.test_indices_file)
    for p, name in [(train_idx_path, "train"), (val_idx_path, "val"), (test_idx_path, "test")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} indices file not found: {p}")

    train_indices = load_indices_from_file(train_idx_path)
    val_indices = load_indices_from_file(val_idx_path)
    test_indices = load_indices_from_file(test_idx_path)

    dataset = GenericPyGDataset(root=str(data_root), processed_dir=args.processed_dir)
    train_loader, val_loader, _ = create_data_loaders(
        dataset, train_indices, val_indices, test_indices, batch_size=args.batch_size
    )
    print(f"Dataset: {len(dataset)} samples; train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    gine_config = GineConfig(
        in_channels=args.in_channels,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        out_classes=args.num_classes,
    )
    save_path = Path(args.save_path) if args.save_path else data_root / "best_gnn.pt"
    run_training(
        train_loader,
        val_loader,
        gine_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_path=save_path,
        label_attr=args.label_attr,
    )


if __name__ == "__main__":
    main()
