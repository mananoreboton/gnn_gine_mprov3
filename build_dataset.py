"""
Convenience entry point: build MPro v3 PyG dataset.
Delegates to mpro.build_dataset. For other datasets, use the dataset-specific build script.
Usage: uv run python build_dataset.py [--data_root /path] [--dataset_name processed_pyg]
"""

from mpro.build_dataset import main

if __name__ == "__main__":
    main()
