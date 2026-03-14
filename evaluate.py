"""
Classify the test set on the trained GNN.
Loads a saved checkpoint from the latest results/trainings/<timestamp>/ and reports test accuracy.
Saves per-sample results to results/classifications/<timestamp>/ for report generation.
Usage:
  uv run python evaluate.py [--data_root /path/to/snapshot] [--checkpoint best_gnn.pt] [--fold_index 0]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_ROOT,
    RESULTS_TRAININGS,
    RESULTS_CLASSIFICATIONS,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_VAL_SPLIT_FILE,
    DEFAULT_TEST_SPLIT_FILE,
    SplitConfig,
)
from evaluation import evaluate_test_with_predictions, print_test_report
from gine_config import GineConfig
from loaders import create_data_loaders
from utils import RunLogger, get_latest_timestamp_dir, run_timestamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GNN on the test set (classification, run independently of training)."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to raw MPro snapshot (Splits/); default: config.DEFAULT_DATA_ROOT",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=f"Root for outputs (default: {DEFAULT_RESULTS_ROOT}); uses latest trainings/ and datasets/, writes to classifications/<timestamp>/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_gnn.pt",
        help="Checkpoint filename (default: best_gnn.pt); loaded from latest results_root/trainings/<timestamp>/.",
    )
    parser.add_argument(
        "--train_split_file",
        type=str,
        default=DEFAULT_TRAIN_SPLIT_FILE,
        help=f"Train split file in Splits/ (default: {DEFAULT_TRAIN_SPLIT_FILE})",
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default=DEFAULT_VAL_SPLIT_FILE,
        help=f"Val split file in Splits/ (default: {DEFAULT_VAL_SPLIT_FILE})",
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=DEFAULT_TEST_SPLIT_FILE,
        help=f"Test split file in Splits/ (default: {DEFAULT_TEST_SPLIT_FILE})",
    )
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--fold_index", type=int, default=0, help="Which fold to use (0 .. num_folds-1)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--hidden", type=int, default=64, help="Must match trained model")
    parser.add_argument("--num_layers", type=int, default=3, help="Must match trained model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Must match trained model")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes (default: 3)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root or DEFAULT_DATA_ROOT)
    results_root = Path(args.results_root or DEFAULT_RESULTS_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    trainings_base = results_root / RESULTS_TRAININGS
    latest_training = get_latest_timestamp_dir(trainings_base)
    if latest_training is None:
        raise FileNotFoundError(f"No training run found under {trainings_base}. Run train.py first.")
    checkpoint_path = latest_training / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_base = results_root / "datasets"
    latest_dataset = get_latest_timestamp_dir(dataset_base)
    if latest_dataset is None or not (latest_dataset / "data.pt").exists():
        raise FileNotFoundError(f"No dataset found under {dataset_base}. Run build_dataset.py first.")
    dataset_name = latest_dataset.name

    split_config = SplitConfig(
        train_file=args.train_split_file,
        val_file=args.val_split_file,
        test_file=args.test_split_file,
        num_folds=args.num_folds,
        fold_index=args.fold_index,
        dataset_name=dataset_name,
    )
    gine_config = GineConfig(
        in_channels=4,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        out_classes=args.num_classes,
    )

    _, _, test_loader = create_data_loaders(
        dataset_base, data_root, split_config, batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gine_config.build().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))

    test_metrics, results = evaluate_test_with_predictions(model, test_loader, device)

    ts = run_timestamp()
    out_dir = results_root / RESULTS_CLASSIFICATIONS / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "evaluate.log"
    results_path = out_dir / "evaluation_results.json"

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root": str(data_root.resolve()),
        "results_root": str(results_root.resolve()),
        "dataset_name": dataset_name,
        "fold_index": split_config.fold_index,
        "num_folds": split_config.num_folds,
        "accuracy": test_metrics.accuracy,
        "results": [
            {"pdb_id": pdb_id, "real_category": real, "predicted_category": pred}
            for pdb_id, real, pred in results
        ],
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with RunLogger(log_path) as log:
        log.log(f"Checkpoint: {checkpoint_path}")
        log.log(f"Dataset: {dataset_base / dataset_name}")
        log.log(f"Test set size: {len(test_loader.dataset)}")
        log.log(f"Test accuracy (Category): {test_metrics.accuracy:.4f}")
        print_test_report(test_metrics)
        log.log(f"Results saved to {results_path}")
        log.log(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
