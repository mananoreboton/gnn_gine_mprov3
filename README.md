# GNN training for MPro Version 3 data

Python pipeline to train a Graph Neural Network on the MPro-URV Version 3 snapshot for **pIC50 regression** and optional **3-class classification** (Category: low / medium / high potency).

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

## Usage

From the project root (with uv, use `uv run` so the virtualenv is used automatically):

```bash
# Default: data root = ../MPro-URV_Version3_snapshot
uv run python train.py

# Or without uv (after uv sync or pip install):
python train.py

# Custom data root
uv run python train.py --data_root /path/to/MPro-URV_Version3_snapshot

# Use predefined train/val/test from Splits folder (5 folds by default)
uv run python train.py --use_splits

# Use a specific fold (0..4) and custom number of folds
uv run python train.py --use_splits --num_folds 5 --fold_index 2

# Three-file format: train/val/test split in separate files (each with num_folds lists)
uv run python train.py --use_splits --split_file train_index_folder.txt --val_split_file val_index_folder.txt --test_split_file test_index_folder.txt --num_folds 5 --fold_index 0

# Regression only (no Category loss)
uv run python train.py --no_classification

# Hyperparameters
uv run python train.py --epochs 150 --batch_size 16 --hidden 128 --lr 5e-4
```

First run builds the PyG dataset from SDFs (with edge attributes for GINE) and saves it under `MPro-URV_Version3_snapshot/processed_pyg/`. The best model is saved as `best_gnn.pt` in the data root. If you had a previous run without edge features, delete the `processed_pyg` folder so the dataset is rebuilt with `edge_attr`.

## Layout

- `dataset.py`: Load SDF → PyG `Data` (x, edge_index, pIC50, category); `MProV3Dataset`; split helpers.
- `model.py`: **GINE** (Graph Isomorphism Network with Edge features) with bond-type edge attributes, global mean pool, and two heads (regression + classification).
- `train.py`: Training loop, MSE for pIC50 and optional CE for Category; reports RMSE and accuracy.
- `config.py`: Default paths and hyperparameters.
