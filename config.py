"""Default paths and hyperparameters for GNN training on MPro Version 3."""

from pathlib import Path

# Default data root: MPro-URV_Version3_snapshot (sibling of gnn_version3 folder)
DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parent.parent / "MPro-URV_Version3_snapshot")

# Model
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
DROPOUT = 0.2
IN_CHANNELS = 4  # x, y, z, atomic number

# Training
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
SEED = 42
USE_CLASSIFICATION = True  # Category 3-class loss
USE_SPLITS = False  # Use predefined train/val/test from Splits folder
