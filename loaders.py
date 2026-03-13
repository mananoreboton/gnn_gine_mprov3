"""
Generic data loaders: collate and factory for train/val/test DataLoaders.
Works with any PyG dataset and index lists; no dataset-specific logic.
"""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


def collate_batch(batch: List) -> Batch:
    """Collate a list of PyG Data into a single Batch (attributes are stacked by PyG)."""
    return Batch.from_data_list([b for b in batch])


def create_data_loaders(
    dataset: Dataset,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders from a dataset and three index lists.
    Indices can be lists or 1D tensors.
    """
    train_idx = train_indices.tolist() if isinstance(train_indices, torch.Tensor) else train_indices
    val_idx = val_indices.tolist() if isinstance(val_indices, torch.Tensor) else val_indices
    test_idx = test_indices.tolist() if isinstance(test_indices, torch.Tensor) else test_indices

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

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
