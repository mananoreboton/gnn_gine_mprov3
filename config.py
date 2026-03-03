"""Default paths and hyperparameters for GNN training on MPro Version 3."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Default data root: MPro-URV_Version3_snapshot (sibling of gnn_version3 folder)
DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parent.parent / "MPro-URV_Version3_snapshot")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for train/val/test split (folds and split files)."""

    use_splits: bool = False
    split_file: str = "train_index_folder.txt"
    val_split_file: Optional[str] = None
    test_split_file: Optional[str] = None
    num_folds: int = 5
    fold_index: int = 0
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the training run (optimizer, epochs, loss)."""

    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    use_classification: bool = True
    classification_loss_weight: float = 0.5


# Legacy constants (for backwards compatibility or quick scripts)
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
DROPOUT = 0.2
IN_CHANNELS = 4
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
SEED = 42
USE_CLASSIFICATION = True
USE_SPLITS = False
NUM_FOLDS = 5
FOLD_INDEX = 0
