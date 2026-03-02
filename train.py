"""
Train GNN on MPro Version 3: regression (pIC50) and optional classification (Category).
Usage:
  python train.py --data_root /path/to/MPro-URV_Version3_snapshot [--use_splits] [--epochs 100]
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from dataset import MProV3Dataset, get_train_val_test_indices
from model import get_model


def collate_batch(batch):
    """Collate so that pIC50 and category are stacked and batch vector is set."""
    from torch_geometric.data import Batch
    data_batch = Batch.from_data_list([b for b in batch])
    pIC50 = torch.cat([b.pIC50 for b in batch], dim=0)
    category = torch.cat([b.category for b in batch], dim=0)
    return data_batch, pIC50, category


def train_one_epoch(model, loader, optimizer, device, criterion_mse, criterion_ce, use_cls):
    model.train()
    total_loss = 0.0
    for data_batch, pIC50, category in loader:
        data_batch = data_batch.to(device)
        pIC50 = pIC50.to(device)
        category = category.to(device).squeeze(-1)
        optimizer.zero_grad()
        edge_attr = getattr(data_batch, "edge_attr", None)
        pred_pIC50, logits = model(
            data_batch.x,
            data_batch.edge_index,
            data_batch.batch,
            edge_attr,
        )
        loss = criterion_mse(pred_pIC50.squeeze(-1), pIC50.squeeze(-1))
        if use_cls and logits is not None:
            loss = loss + 0.5 * criterion_ce(logits, category)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, use_cls=False):
    model.eval()
    mse_sum = 0.0
    n = 0
    correct_cls = 0
    total_cls = 0
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
        if use_cls and logits is not None:
            pred_cls = logits.argmax(dim=1)
            correct_cls += (pred_cls == category).sum().item()
            total_cls += category.size(0)
    rmse = (mse_sum / n) ** 0.5 if n else 0.0
    acc = correct_cls / total_cls if total_cls else 0.0
    return rmse, acc


def main():
    parser = argparse.ArgumentParser(description="Train GNN on MPro Version 3")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to MPro-URV_Version3_snapshot (default: ../MPro-URV_Version3_snapshot)",
    )
    parser.add_argument("--use_splits", action="store_true", help="Use train/val/test from Splits folder")
    parser.add_argument("--split_file", type=str, default="train_index_folder.txt")
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
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = str(Path(__file__).resolve().parent.parent / "MPro-URV_Version3_snapshot")
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    torch.manual_seed(args.seed)

    # Dataset (processes SDFs and saves to processed_pyg if needed)
    dataset = MProV3Dataset(
        root=str(data_root),
        use_splits=False,
    )
    n = len(dataset)
    print(f"Dataset size: {n}")

    # Splits
    train_idx, val_idx, test_idx = get_train_val_test_indices(
        n,
        data_root,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed,
        use_splits=args.use_splits,
        split_file=args.split_file,
    )
    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())
    test_dataset = Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 4
    model = get_model(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        out_classes=3 if args.classification else None,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    best_val_rmse = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            criterion_mse, criterion_ce, args.classification,
        )
        val_rmse, val_acc = evaluate(model, val_loader, device, args.classification)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), data_root / "best_gnn.pt")
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_rmse={val_rmse:.4f}"
                + (f"  val_acc={val_acc:.4f}" if args.classification else "")
            )

    model.load_state_dict(torch.load(data_root / "best_gnn.pt"))
    test_rmse, test_acc = evaluate(model, test_loader, device, args.classification)
    print(f"Test RMSE (pIC50): {test_rmse:.4f}")
    if args.classification:
        print(f"Test accuracy (Category): {test_acc:.4f}")


if __name__ == "__main__":
    main()
