"""
Data loaders: collate function and factory for train/val/test DataLoaders.
"""

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from dataset import MProV3Dataset, get_train_val_test_indices
from config import SplitConfig


def collate_batch(
    batch: List,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate so that pIC50 and category are stacked and batch vector is set."""
    from torch_geometric.data import Batch

    data_batch = Batch.from_data_list([b for b in batch])
    pIC50 = torch.cat([b.pIC50 for b in batch], dim=0)
    category = torch.cat([b.category for b in batch], dim=0)
    return data_batch, pIC50, category


def create_data_loaders(
    data_root: Path,
    split_config: SplitConfig,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders from the full dataset.
    Uses split config for indices; same collate for all.
    """
    dataset = MProV3Dataset(root=str(data_root), use_splits=False)
    n = len(dataset)
    train_idx, val_idx, test_idx = get_train_val_test_indices(
        n,
        data_root,
        val_ratio=split_config.val_ratio,
        test_ratio=split_config.test_ratio,
        seed=split_config.seed,
        use_splits=split_config.use_splits,
        split_file=split_config.split_file,
        val_split_file=split_config.val_split_file,
        test_split_file=split_config.test_split_file,
        num_folds=split_config.num_folds,
        fold_index=split_config.fold_index,
    )
    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())
    test_dataset = Subset(dataset, test_idx.tolist())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader, test_loader
