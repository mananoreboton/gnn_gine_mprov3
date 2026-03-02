# GNN training for MPro Version 3 data

Python pipeline to train a Graph Neural Network on the MPro-URV Version 3 snapshot for **pIC50 regression** and optional **3-class classification** (Category: low / medium / high potency).

## Data

- **Graphs**: One graph per ligand from `Ligand/Ligand_SDF/*.sdf`. Node features: 3D coordinates (x, y, z) and atomic number. Edges: bonds; edge features: bond type (single=1, double=2, triple=3, aromatic=1.5) for GINE.
- **Labels**: `Info.csv` provides `pIC50` and `Category` (-1: pIC50&lt;5.5, 0: 5.5≤pIC50&lt;6.5, 1: pIC50≥6.5). Category is mapped to classes 0, 1, 2.

## Setup

```bash
cd gnn_version3
pip install -r requirements.txt
```

Requires: PyTorch, PyTorch Geometric, RDKit, pandas, numpy, scikit-learn.

## Usage

From the project root or from `gnn_version3`:

```bash
# Default: data root = ../MPro-URV_Version3_snapshot
python -m gnn_version3.train

# Custom data root
python -m gnn_version3.train --data_root /path/to/MPro-URV_Version3_snapshot

# Use predefined train/val/test from Splits/train_index_folder.txt (first 3 folds)
python -m gnn_version3.train --use_splits

# Regression only (no Category loss)
python -m gnn_version3.train --no_classification

# Hyperparameters
python -m gnn_version3.train --epochs 150 --batch_size 16 --hidden 128 --lr 5e-4
```

First run builds the PyG dataset from SDFs (with edge attributes for GINE) and saves it under `MPro-URV_Version3_snapshot/processed_pyg/`. The best model is saved as `best_gnn.pt` in the data root. If you had a previous run without edge features, delete the `processed_pyg` folder so the dataset is rebuilt with `edge_attr`.

## Layout

- `dataset.py`: Load SDF → PyG `Data` (x, edge_index, pIC50, category); `MProV3Dataset`; split helpers.
- `model.py`: **GINE** (Graph Isomorphism Network with Edge features) with bond-type edge attributes, global mean pool, and two heads (regression + classification).
- `train.py`: Training loop, MSE for pIC50 and optional CE for Category; reports RMSE and accuracy.
- `config.py`: Default paths and hyperparameters.
