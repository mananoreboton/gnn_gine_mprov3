"""
Generic PyG dataset loader: load a pre-built dataset from a path (data.pt).
Works with any dataset produced in PyG InMemoryDataset format; no dataset-specific logic.
"""

from pathlib import Path
from typing import List

from torch_geometric.data import InMemoryDataset


def load_indices_from_file(path: Path) -> List[int]:
    """Load integer indices from a file (one index per line)."""
    text = path.read_text().strip()
    if not text:
        return []
    return [int(line.strip()) for line in text.splitlines() if line.strip()]


class GenericPyGDataset(InMemoryDataset):
    """
    Load a pre-built PyG dataset from root / processed_dir / data.pt.
    Expects PyG InMemoryDataset.save() format. Use for any dataset-agnostic training/evaluation.
    """

    def __init__(
        self,
        root: str,
        processed_dir: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        root_path = Path(root)
        self._processed_dir_name = processed_dir
        data_path = root_path / processed_dir / "data.pt"
        if not data_path.exists():
            raise FileNotFoundError(
                f"PyG dataset not found at {data_path}. "
                "Build the dataset first (e.g. run the dataset-specific build script)."
            )
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / self._processed_dir_name)

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / self._processed_dir_name)

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def process(self):
        raise FileNotFoundError(
            f"Processed file not found under {self.processed_dir}. "
            "Build the dataset first using the dataset-specific build script."
        )
