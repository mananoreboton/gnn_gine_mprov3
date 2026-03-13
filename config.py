"""Generic configuration: training hyperparameters. Dataset-specific config lives in dataset packages (e.g. mpro.config)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a training run."""

    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42


# Generic defaults for quick scripts
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
DROPOUT = 0.2
IN_CHANNELS = 4
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
SEED = 42
