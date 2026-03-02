"""
MPro Version 3 dataset: load SDF ligands and build PyTorch Geometric graphs.
Node features: (x, y, z, atomic_number). Edges from bonds.
Labels: pIC50 (regression) and Category (classification: -1, 0, 1 -> 0, 1, 2).
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


# Atomic number lookup (common elements in MPro ligands)
ATOMIC_NUM = {
    "C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17, "Br": 35,
    "I": 53, "P": 15, "B": 5, "H": 1,
}


def _bond_type_to_scalar(bond) -> float:
    """Map RDKit bond type to scalar for GINE edge_attr. Single=1, Double=2, Triple=3, Aromatic=1.5."""
    from rdkit.Chem import BondType
    bt = bond.GetBondType()
    if bt == BondType.SINGLE:
        return 1.0
    if bt == BondType.DOUBLE:
        return 2.0
    if bt == BondType.TRIPLE:
        return 3.0
    if bt == BondType.AROMATIC:
        return 1.5
    return 1.0


def sdf_to_graph(sdf_path: Path) -> Optional[Data]:
    """Load one SDF and return a PyG Data with x (N,4): [x,y,z, atomic_num], edge_index (2,E), edge_attr (E,1) for GINE."""
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        return None

    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    pos = conf.GetPositions()  # (n, 3)
    node_feats = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        sym = atom.GetSymbol()
        anum = ATOMIC_NUM.get(sym, 6)  # default C if unknown
        node_feats.append([pos[i, 0], pos[i, 1], pos[i, 2], float(anum)])
    x = torch.tensor(node_feats, dtype=torch.float32)

    edge_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _bond_type_to_scalar(bond)
        edge_list.append([u, v])
        edge_attr_list.append([bt])
        edge_list.append([v, u])
        edge_attr_list.append([bt])
    if len(edge_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


def load_activity_and_category(
    data_root: Path,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Load pIC50 and Category from Info.csv. Return dicts PDB_ID -> value."""
    info_path = data_root / "Info.csv"
    df = pd.read_csv(info_path, sep=";")
    pIC50 = dict(zip(df["PDB_ID"].astype(str), df["pIC50"].astype(float)))
    # Category: -1 -> 0, 0 -> 1, 1 -> 2 for class indices
    cat_map = {-1: 0, 0: 1, 1: 2}
    category = {}
    for _, row in df.iterrows():
        c = int(row["Category"])
        category[str(row["PDB_ID"])] = cat_map.get(c, 0)
    return pIC50, category


def load_splits(data_root: Path, split_file: str) -> List[List[str]]:
    """Parse Splits file (Python list of lists of PDB IDs). Returns list of folds."""
    path = data_root / "Splits" / split_file
    text = path.read_text()
    # Parse as Python literal: list of lists
    import ast
    folds = ast.literal_eval(text)
    if isinstance(folds, list) and len(folds) > 0:
        if isinstance(folds[0], list):
            return folds
        return [folds]
    return []


class MProV3Dataset(InMemoryDataset):
    """
    In-memory PyG dataset from MPro-URV Version 3 snapshot.
    Root should point to MPro-URV_Version3_snapshot (parent of Ligand/, Info.csv, etc.).
    """

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_splits: bool = False,
        split_file: str = "train_index_folder.txt",
        fold_index: int = 0,
    ):
        self._data_root = Path(root)
        self.use_splits = use_splits
        self.split_file = split_file
        self.fold_index = fold_index
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return str(self._data_root)

    @property
    def processed_dir(self) -> str:
        return str(self._data_root / "processed_pyg")

    @property
    def raw_file_names(self) -> List[str]:
        return ["Info.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def process(self):
        sdf_dir = self._data_root / "Ligand" / "Ligand_SDF"
        pIC50_dict, category_dict = load_activity_and_category(self._data_root)

        if self.use_splits:
            folds = load_splits(self._data_root, self.split_file)
            if folds and self.fold_index < len(folds):
                pdb_ids = folds[self.fold_index]
            else:
                pdb_ids = list(pIC50_dict.keys())
        else:
            pdb_ids = sorted(pIC50_dict.keys())

        data_list = []
        for pdb_id in tqdm(pdb_ids, desc="Loading graphs"):
            sdf_path = sdf_dir / f"{pdb_id}_ligand.sdf"
            if not sdf_path.exists():
                continue
            g = sdf_to_graph(sdf_path)
            if g is None:
                continue
            if pdb_id not in pIC50_dict:
                continue
            g.pIC50 = torch.tensor([pIC50_dict[pdb_id]], dtype=torch.float32)
            g.category = torch.tensor([category_dict.get(pdb_id, 0)], dtype=torch.long)
            g.pdb_id = pdb_id
            data_list.append(g)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        self.save(data_list, self.processed_paths[0])


def get_train_val_test_indices(
    n: int,
    data_root: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    use_splits: bool = False,
    split_file: str = "train_index_folder.txt",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return train, val, test indices. If use_splits, parse Splits and map PDB order to indices.
    Otherwise random split.
    """
    if use_splits:
        folds = load_splits(data_root, split_file)
        if len(folds) >= 3:
            train_ids = set(folds[0])
            val_ids = set(folds[1])
            test_ids = set(folds[2])
        else:
            use_splits = False
    if not use_splits:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_val - n_test
        return (
            torch.tensor(perm[:n_train]),
            torch.tensor(perm[n_train : n_train + n_val]),
            torch.tensor(perm[n_train + n_val :]),
        )

    # Build PDB -> index from dataset order (same as in MProV3Dataset process: sorted pdb_ids)
    pIC50_dict, _ = load_activity_and_category(data_root)
    pdb_order = sorted(pIC50_dict.keys())
    pdb_to_idx = {p: i for i, p in enumerate(pdb_order)}

    train_idx = [pdb_to_idx[p] for p in train_ids if p in pdb_to_idx]
    val_idx = [pdb_to_idx[p] for p in val_ids if p in pdb_to_idx]
    test_idx = [pdb_to_idx[p] for p in test_ids if p in pdb_to_idx]
    return (
        torch.tensor(train_idx),
        torch.tensor(val_idx),
        torch.tensor(test_idx),
    )
