"""MPro Version 3 default paths and split configuration."""

from dataclasses import dataclass
from pathlib import Path

# Default data root: MPro snapshot (sibling of project folder)
DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parent.parent.parent / "MPro-URV_Version3_snapshot")
# Default name of the PyG dataset folder under data_root (contains data.pt)
DEFAULT_PYG_DATASET_NAME = "processed_pyg"

# Default split file names (in data_root/Splits/)
DEFAULT_TRAIN_SPLIT_FILE = "train_index_folder.txt"
DEFAULT_VAL_SPLIT_FILE = "valid_index_folder.txt"
DEFAULT_TEST_SPLIT_FILE = "test_index_folder.txt"


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for MPro train/val/test split: three files and folds."""

    train_file: str = DEFAULT_TRAIN_SPLIT_FILE
    val_file: str = DEFAULT_VAL_SPLIT_FILE
    test_file: str = DEFAULT_TEST_SPLIT_FILE
    num_folds: int = 5
    fold_index: int = 0
    dataset_name: str = DEFAULT_PYG_DATASET_NAME
