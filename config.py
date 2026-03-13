"""Default paths and hyperparameters for GNN training on MPro Version 3."""

from dataclasses import dataclass
from pathlib import Path

# Default data root: MPro-URV_Version3_snapshot (sibling of gnn_version3 folder)
DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parent.parent / "MPro-URV_Version3_snapshot")
# Default name of the PyG dataset folder under data_root (contains data.pt)
DEFAULT_PYG_DATASET_NAME = "processed_pyg"


# Default split file names (in data_root/Splits/)
DEFAULT_TRAIN_SPLIT_FILE = "train_index_folder.txt"
DEFAULT_VAL_SPLIT_FILE = "valid_index_folder.txt"
DEFAULT_TEST_SPLIT_FILE = "test_index_folder.txt"


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for train/val/test split: three files and folds."""

    train_file: str = DEFAULT_TRAIN_SPLIT_FILE
    val_file: str = DEFAULT_VAL_SPLIT_FILE
    test_file: str = DEFAULT_TEST_SPLIT_FILE
    num_folds: int = 5
    fold_index: int = 0
    dataset_name: str = DEFAULT_PYG_DATASET_NAME


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the training run."""

    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42


# Legacy constants (for backwards compatibility or quick scripts)
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
DROPOUT = 0.2
IN_CHANNELS = 4
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
SEED = 42
USE_SPLITS = False
NUM_FOLDS = 5
FOLD_INDEX = 0
