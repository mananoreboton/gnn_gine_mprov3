"""
Build the MPro v3 PyG dataset from SDFs and Info.csv. Run once before training.
Usage: uv run python -m mpro.build_dataset [--data_root /path] [--dataset_name processed_pyg]
"""

import argparse
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset

from mpro.config import DEFAULT_DATA_ROOT, DEFAULT_PYG_DATASET_NAME
from mpro.dataset import load_activity_and_category, sdf_to_graph
from tqdm import tqdm


def build_and_save_pyg_dataset(
    data_root: Path,
    dataset_name: str = DEFAULT_PYG_DATASET_NAME,
) -> Path:
    """
    Build PyG graph list from MPro SDFs and Info.csv; save to data_root/dataset_name/data.pt.
    Also writes pdb_order.txt for split index mapping. Returns the path to data.pt.
    """
    sdf_dir = data_root / "Ligand" / "Ligand_SDF"
    if not sdf_dir.exists():
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")
    pIC50_dict, category_dict = load_activity_and_category(data_root)
    pdb_ids = sorted(pIC50_dict.keys())
    data_list = []
    dataset_pdb_order = []
    for pdb_id in tqdm(pdb_ids, desc="Building PyG dataset"):
        sdf_path = sdf_dir / f"{pdb_id}_ligand.sdf"
        if not sdf_path.exists():
            continue
        g = sdf_to_graph(sdf_path)
        if g is None:
            continue
        g.pIC50 = torch.tensor([pIC50_dict[pdb_id]], dtype=torch.float32)
        g.category = torch.tensor([category_dict.get(pdb_id, 0)], dtype=torch.long)
        g.pdb_id = pdb_id
        data_list.append(g)
        dataset_pdb_order.append(pdb_id)
    out_dir = data_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.pt"
    InMemoryDataset.save(data_list, str(out_path))
    pdb_order_path = out_dir / "pdb_order.txt"
    pdb_order_path.write_text("\n".join(dataset_pdb_order))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MPro v3 PyG dataset from SDFs and Info.csv. Run before training."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=f"Path to MPro snapshot (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_PYG_DATASET_NAME,
        help=f"Name of the PyG dataset folder under data_root (default: {DEFAULT_PYG_DATASET_NAME})",
    )
    args = parser.parse_args()
    data_root = Path(args.data_root or DEFAULT_DATA_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    path = build_and_save_pyg_dataset(data_root, args.dataset_name)
    print(f"PyG dataset saved to {path}")


if __name__ == "__main__":
    main()
