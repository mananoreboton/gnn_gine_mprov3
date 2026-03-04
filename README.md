# GNN training for MPro Version 3 data

Python pipeline to train a Graph Neural Network on the MPro-URV Version 3 snapshot for **pIC50 regression** and optional **3-class classification** (Category: low / medium / high potency). The codebase is split into configuration, data loading, GINE model, and separate training, validation, and testing logic.

## Data

- **Graphs**: One graph per ligand from `Ligand/Ligand_SDF/*.sdf`. Node features: 3D coordinates (x, y, z) and atomic number. Edges: bonds; edge features: bond type (single=1, double=2, triple=3, aromatic=1.5) for GINE.
- **Labels**: `Info.csv` provides `pIC50` and `Category` (-1: pIC50&lt;5.5, 0: 5.5≤pIC50&lt;6.5, 1: pIC50≥6.5). Category is mapped to classes 0, 1, 2.

## Setup

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
cd gnn_gine_mprov3
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

Requires: PyTorch, PyTorch Geometric, RDKit, pandas, numpy, scikit-learn.

---

## Usage

### 1. Build the PyG dataset (required once)

Train/val/test loaders use a **pre-built** PyG dataset. Create it from SDFs and `Info.csv` before training:

```bash
# Default: data_root = ../MPro-URV_Version3_snapshot, dataset saved as processed_pyg/
uv run python build_dataset.py

# Custom data root and dataset name
uv run python build_dataset.py --data_root /path/to/MPro-URV_Version3_snapshot --dataset_name processed_pyg
```

The dataset is written to `data_root/<dataset_name>/data.pt`. If you skip this step, training will exit with an error telling you to run `build_dataset.py` first.

### 2. Train (train.py)

Train/val/test splits are **always** read from **three files** in `data_root/Splits/`:

- **Default file names**: `train_index_folder.txt`, `valid_index_folder.txt`, `test_index_folder.txt`
- Each file must contain **num_folds** lists of PDB IDs (one list per fold). Default **num_folds** is **5**.

```bash
# Default: data root, default split files, 5 folds, fold 0
uv run python train.py

# Custom data root
uv run python train.py --data_root /path/to/MPro-URV_Version3_snapshot
```

#### PyG dataset and split files

```bash
# Name of the PyG dataset folder to load (must exist; created by build_dataset.py)
uv run python train.py --dataset_name processed_pyg

# Override split file names (defaults: train_index_folder.txt, valid_index_folder.txt, test_index_folder.txt)
uv run python train.py --train_split_file train_index_folder.txt --val_split_file valid_index_folder.txt --test_split_file test_index_folder.txt

# Number of folds (default: 5) and which fold to use (0 .. num_folds-1)
uv run python train.py --num_folds 5 --fold_index 2
```

#### Training options

```bash
# Epochs, batch size, learning rate, seed
uv run python train.py --epochs 150 --batch_size 16 --lr 5e-4 --seed 42

# Regression only (no Category classification loss)
uv run python train.py --no_classification
```

#### GINE model (architecture)

```bash
# Hidden size, depth, dropout
uv run python train.py --hidden 128 --num_layers 4 --dropout 0.2
```

#### Full example

```bash
uv run python build_dataset.py --data_root /path/to/MPro-URV_Version3_snapshot
uv run python train.py \
  --data_root /path/to/MPro-URV_Version3_snapshot \
  --dataset_name processed_pyg \
  --num_folds 5 --fold_index 0 \
  --epochs 100 --batch_size 32 --lr 1e-3 \
  --hidden 64 --num_layers 3 --dropout 0.2 \
  --classification --seed 42
```

The best model (by validation RMSE) is saved as `best_gnn.pt` in the data root.

---

### Programmatic use

You can reuse configs, loaders, and train/val/test logic in your own scripts.

#### Configuration

- **`config.SplitConfig`**: train/val/test file names (`train_file`, `val_file`, `test_file`), `num_folds`, `fold_index`, `dataset_name` (PyG dataset folder).
- **`config.TrainingConfig`**: training run (`epochs`, `batch_size`, `lr`, `seed`, `use_classification`, `classification_loss_weight`).
- **`gine_config.GineConfig`**: GINE architecture (`in_channels`, `hidden_channels`, `num_layers`, `dropout`, etc.). Call `.build()` to get an `MProGNN` instance.

#### Data loaders

- **`loaders.collate_batch(batch)`**: collate list of PyG graphs into a batch plus pIC50 and category tensors.
- **`loaders.create_data_loaders(data_root, split_config, batch_size=32)`**: loads the PyG dataset from `data_root/split_config.dataset_name` (must exist) and returns `(train_loader, val_loader, test_loader)` using the three split files and `fold_index`.

#### Training

- **`training.train_one_epoch(model, loader, optimizer, device, criterion_mse, criterion_ce, use_classification, classification_loss_weight)`**: one training epoch; returns mean loss.

#### Validation

- **`validation.evaluate_validation(model, loader, device, use_classification=False)`**: returns **`ValidationMetrics`** (`rmse`, optional `accuracy`).

#### Testing

- **`testing.evaluate_test(model, loader, device, use_classification=False)`**: returns **`TestMetrics`** (`rmse`, optional `accuracy`).
- **`testing.print_test_report(metrics, use_classification=False)`**: prints test RMSE and optional accuracy.

#### Example script

```python
from pathlib import Path
import torch
from config import SplitConfig, TrainingConfig
from gine_config import GineConfig
from loaders import create_data_loaders
from training import train_one_epoch
from validation import evaluate_validation
from testing import evaluate_test, print_test_report

data_root = Path("/path/to/MPro-URV_Version3_snapshot")
# PyG dataset must exist at data_root/processed_pyg/data.pt (run build_dataset.py first)
split_config = SplitConfig(num_folds=5, fold_index=0, dataset_name="processed_pyg")
training_config = TrainingConfig(epochs=50, batch_size=32, lr=1e-3, use_classification=True)
gine_config = GineConfig(hidden_channels=64, num_layers=3, dropout=0.2, out_classes=3)

train_loader, val_loader, test_loader = create_data_loaders(data_root, split_config, training_config.batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = gine_config.build().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)
criterion_mse = torch.nn.MSELoss()
criterion_ce = torch.nn.CrossEntropyLoss()

# Training loop (simplified)
for epoch in range(1, training_config.epochs + 1):
    train_one_epoch(model, train_loader, optimizer, device, criterion_mse, criterion_ce,
                    training_config.use_classification, training_config.classification_loss_weight)
    val_metrics = evaluate_validation(model, val_loader, device, training_config.use_classification)
    # ... save best model, etc.

model.load_state_dict(torch.load(data_root / "best_gnn.pt"))
test_metrics = evaluate_test(model, test_loader, device, training_config.use_classification)
print_test_report(test_metrics, training_config.use_classification)
```

---

## Layout

| File | Role |
|------|------|
| **config.py** | Default paths, default split file names, `DEFAULT_PYG_DATASET_NAME`; `SplitConfig`, `TrainingConfig`. |
| **gine_config.py** | GINE config: `GineConfig` dataclass and `.build()` → `MProGNN`. |
| **model.py** | GINE model logic: `MProGNN`. |
| **dataset.py** | Helpers: `sdf_to_graph`, `load_activity_and_category`; `load_splits` (three files); `get_train_val_test_indices`; `MProV3Dataset` (loads pre-built PyG dataset, errors if missing). |
| **build_dataset.py** | Builds PyG dataset from SDFs and saves to `data_root/<dataset_name>/data.pt`. Run once before training. |
| **loaders.py** | `collate_batch`, `create_data_loaders` (require existing PyG dataset). |
| **training.py** | Training logic: `train_one_epoch`. |
| **validation.py** | Validation: `evaluate_validation`, `ValidationMetrics`. |
| **testing.py** | Testing: `evaluate_test`, `TestMetrics`, `print_test_report`. |
| **train.py** | CLI entry point: parses args, builds configs and loaders, runs train/val loop and test. |
