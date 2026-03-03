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

### Command-line (train.py)

From the project root, run (with uv use `uv run` so the virtualenv is used automatically):

```bash
# Default: data root = ../MPro-URV_Version3_snapshot, random train/val/test split
uv run python train.py

# Custom data root
uv run python train.py --data_root /path/to/MPro-URV_Version3_snapshot
```

#### Data and split (folds)

- **Random split** (default): train/val/test are chosen randomly with 80/10/10; use `--seed` for reproducibility.
- **Predefined splits from Splits folder**: use `--use_splits`. You can use either:
  - **Single file**: one file (e.g. `train_index_folder.txt`) containing `3 * num_folds` lists: `[train0, val0, test0, train1, val1, test1, ...]`.
  - **Three files**: one file per role, each with `num_folds` lists; specify all three with `--split_file`, `--val_split_file`, `--test_split_file`.

```bash
# Use predefined splits (5 folds by default), fold 0
uv run python train.py --use_splits

# Choose a specific fold (0 .. num_folds-1) and number of folds
uv run python train.py --use_splits --num_folds 5 --fold_index 2

# Three-file format: train/val/test in separate files
uv run python train.py --use_splits \
  --split_file train_index_folder.txt \
  --val_split_file val_index_folder.txt \
  --test_split_file test_index_folder.txt \
  --num_folds 5 --fold_index 0
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
uv run python train.py \
  --data_root /path/to/MPro-URV_Version3_snapshot \
  --use_splits --num_folds 5 --fold_index 0 \
  --epochs 100 --batch_size 32 --lr 1e-3 \
  --hidden 64 --num_layers 3 --dropout 0.2 \
  --classification --seed 42
```

First run builds the PyG dataset from SDFs and saves it under `data_root/processed_pyg/`. The best model (by validation RMSE) is saved as `best_gnn.pt` in the data root. If you change preprocessing (e.g. edge features), delete `processed_pyg` to force a rebuild.

---

### Programmatic use

You can reuse configs, loaders, and train/val/test logic in your own scripts.

#### Configuration

- **`config.SplitConfig`**: split/fold settings (`use_splits`, split file names, `num_folds`, `fold_index`, `val_ratio`, `test_ratio`, `seed`).
- **`config.TrainingConfig`**: training run (`epochs`, `batch_size`, `lr`, `seed`, `use_classification`, `classification_loss_weight`).
- **`gine_config.GineConfig`**: GINE architecture (`in_channels`, `hidden_channels`, `num_layers`, `dropout`, `out_regression`, `out_classes`, `pool`, `edge_dim`). Call `.build()` to get an `MProGNN` instance.

#### Data loaders

- **`loaders.collate_batch(batch)`**: collate list of PyG graphs into a batch plus pIC50 and category tensors.
- **`loaders.create_data_loaders(data_root, split_config, batch_size=32)`**: returns `(train_loader, val_loader, test_loader)` using `SplitConfig` for indices (random or from Splits folder).

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
split_config = SplitConfig(use_splits=True, num_folds=5, fold_index=0, seed=42)
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
| **config.py** | Default paths and typed configs: `SplitConfig`, `TrainingConfig`. |
| **gine_config.py** | GINE config: `GineConfig` dataclass and `.build()` → `MProGNN`; separates config from model logic. |
| **model.py** | GINE model logic only: `MProGNN` (Graph Isomorphism Network with Edge features). |
| **dataset.py** | SDF → PyG `Data`; `MProV3Dataset`; split/fold loading from Splits folder. |
| **loaders.py** | `collate_batch`, `create_data_loaders` for train/val/test DataLoaders. |
| **training.py** | Training logic: `train_one_epoch`. |
| **validation.py** | Validation: `evaluate_validation`, `ValidationMetrics`. |
| **testing.py** | Testing: `evaluate_test`, `TestMetrics`, `print_test_report`. |
| **train.py** | CLI entry point: parses args, builds configs and loaders, runs train/val loop and test. |
