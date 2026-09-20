"""Train the MLP predictor from eight independently preprocessed Parquet shards."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import torch


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Predictor0916.src.mlp import MLP
from Predictor0916.src.trainer import Trainer
from Predictor0916.src.utils import compute_metrics, ensure_dir, get_logger, save_json, set_seed


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine eight preprocessing shards, split them approximately 6:1:1, "
            "train on one device, and evaluate once on the held-out validation split."
        )
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        required=True,
        help="Root produced by preprocess_8die.py, or its processed/ directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parquet-pattern", default="shard-*.parquet")
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument(
        "--target-quantiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.01, 0.99),
    )
    parser.add_argument("--loss-type", choices=("mae", "soft_label"), default="soft_label")
    parser.add_argument("--lambda-val", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stopping patience on the test split; use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-checkpoint", type=Path)
    return parser


def resolve_data_directories(path: Path) -> tuple[Path, Path | None]:
    """Accept either the preprocessing root or its processed directory."""
    root = path.resolve()
    nested_processed = root / "processed"
    if nested_processed.is_dir():
        input_dir = root / "inputs"
        return nested_processed, input_dir if input_dir.is_dir() else None
    if root.is_dir() and root.name == "processed":
        input_dir = root.parent / "inputs"
        return root, input_dir if input_dir.is_dir() else None
    if root.is_dir():
        return root, None
    raise FileNotFoundError(f"Preprocessed directory does not exist: {path}")


def load_shards(
    preprocessed_dir: Path,
    pattern: str,
    expected_shards: int,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame, list[Path]]:
    """Load all shards and return tensors plus row provenance metadata."""
    if expected_shards <= 0:
        raise ValueError("expected-shards must be positive.")
    processed_dir, input_dir = resolve_data_directories(preprocessed_dir)
    shard_paths = sorted(processed_dir.glob(pattern))
    if len(shard_paths) != expected_shards:
        raise ValueError(
            f"Expected {expected_shards} files matching {pattern!r} in {processed_dir}, "
            f"but found {len(shard_paths)}: {[path.name for path in shard_paths]}"
        )

    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    provenance_frames: list[pd.DataFrame] = []
    feature_dim: int | None = None

    for shard_path in shard_paths:
        frame = pd.read_parquet(
            shard_path,
            columns=["hidden_state", "response_length"],
        )
        if frame.empty:
            raise ValueError(f"Processed shard is empty: {shard_path}")
        shard_features = [np.asarray(value, dtype=np.float32) for value in frame["hidden_state"]]
        shapes = {value.shape for value in shard_features}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise ValueError(
                f"All hidden_state rows in {shard_path} must be equally sized 1D vectors."
            )
        current_dim = int(next(iter(shapes))[0])
        if feature_dim is None:
            feature_dim = current_dim
        elif current_dim != feature_dim:
            raise ValueError(
                f"Hidden dimension mismatch in {shard_path}: {current_dim} != {feature_dim}."
            )
        stacked_features = np.stack(shard_features)
        targets = frame["response_length"].to_numpy(dtype=np.float32)
        if not np.isfinite(stacked_features).all() or not np.isfinite(targets).all():
            raise ValueError(f"Shard contains non-finite values: {shard_path}")
        if (targets < 0).any():
            raise ValueError(f"Shard contains negative response lengths: {shard_path}")
        feature_rows.append(stacked_features)
        target_rows.append(targets)

        provenance = pd.DataFrame(
            {
                "shard_file": shard_path.name,
                "shard_row": np.arange(len(frame), dtype=np.int64),
            }
        )
        if input_dir is not None:
            input_path = input_dir / f"{shard_path.stem}.csv"
            if not input_path.is_file():
                raise FileNotFoundError(
                    f"Missing input sidecar for {shard_path.name}: {input_path}"
                )
            input_frame = pd.read_csv(input_path)
            if len(input_frame) != len(frame):
                raise ValueError(
                    f"Row mismatch between {input_path} ({len(input_frame)}) and "
                    f"{shard_path} ({len(frame)})."
                )
            for column in ("source_split", "source_index", "global_index"):
                if column in input_frame:
                    provenance[column] = input_frame[column].to_numpy()
        provenance_frames.append(provenance)

    features = torch.from_numpy(np.concatenate(feature_rows, axis=0))
    targets = torch.from_numpy(np.concatenate(target_rows, axis=0))
    provenance = pd.concat(provenance_frames, ignore_index=True)
    return features, targets, provenance, shard_paths


def split_indices(sample_count: int, seed: int) -> dict[str, torch.Tensor]:
    """Create a deterministic approximate 6:1:1 train/test/validation split."""
    if sample_count < 8:
        raise ValueError("At least 8 samples are required for a 6:1:1 split.")
    test_size = max(1, sample_count // 8)
    validation_size = max(1, sample_count // 8)
    train_size = sample_count - test_size - validation_size
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(sample_count, generator=generator)
    return {
        "train": permutation[:train_size],
        "test": permutation[train_size : train_size + test_size],
        "validation": permutation[train_size + test_size :],
    }


def select_rows(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tensor.index_select(0, indices).contiguous()


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    if args.device.split(":", maxsplit=1)[0] == "npu":
        try:
            importlib.import_module("torch_npu")
        except ImportError as exc:
            raise ImportError(
                "NPU training requires torch_npu matching the installed PyTorch and CANN versions."
            ) from exc
    low_quantile, high_quantile = args.target_quantiles
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("target-quantiles must satisfy 0 <= LOW < HIGH <= 1.")
    if args.patience < 0:
        raise ValueError("patience must be non-negative.")

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    logger.info("Loading preprocessing shards from %s", args.preprocessed_dir)
    features, targets, provenance, shard_paths = load_shards(
        args.preprocessed_dir,
        args.parquet_pattern,
        args.expected_shards,
    )
    indices = split_indices(len(targets), args.seed)
    split_tensors = {
        name: (
            select_rows(features, split_indices_value),
            select_rows(targets, split_indices_value),
        )
        for name, split_indices_value in indices.items()
    }

    train_features, train_targets = split_tensors["train"]
    test_features, test_targets = split_tensors["test"]
    validation_features, validation_targets = split_tensors["validation"]
    range_low, range_high = np.quantile(
        train_targets.numpy(),
        (low_quantile, high_quantile),
    )
    if range_high <= range_low:
        range_high = range_low + 1.0
    target_range = (float(range_low), float(range_high))

    head = MLP(
        input_dim=int(features.shape[1]),
        num_bins=args.num_bins,
        target_range=target_range,
    )
    if args.load_checkpoint is not None:
        head.load(args.load_checkpoint)
    trainer = Trainer(
        model=head,
        device=args.device,
        loss_type=args.loss_type,
        lambda_val=args.lambda_val,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=None if args.patience == 0 else args.patience,
        seed=args.seed,
    )

    logger.info(
        "Training with train=%d, test=%d, held-out validation=%d",
        len(train_targets),
        len(test_targets),
        len(validation_targets),
    )
    # The test split is used only for epoch-level model selection and early stopping.
    history = trainer.fit(
        train_features,
        train_targets,
        test_features,
        test_targets,
    )
    test_predictions = trainer.predict(test_features).numpy()
    test_metrics = compute_metrics(test_predictions, test_targets.numpy())

    # The validation split remains untouched until training and model selection finish.
    validation_predictions = trainer.predict(validation_features).numpy()
    validation_targets_np = validation_targets.numpy()
    validation_metrics = compute_metrics(validation_predictions, validation_targets_np)

    checkpoint_path = output_dir / "checkpoints" / "best_layers.pt"
    head.save(checkpoint_path)
    assignments = provenance.copy()
    assignments["assigned_split"] = ""
    for name, split_indices_value in indices.items():
        assignments.loc[split_indices_value.numpy(), "assigned_split"] = name
    assignments.to_csv(output_dir / "split_assignments.csv", index=False)

    validation_rows = provenance.iloc[indices["validation"].numpy()].reset_index(drop=True)
    validation_rows.insert(0, "sample_index", np.arange(len(validation_rows)))
    validation_rows["predicted_length"] = validation_predictions
    validation_rows["target_length"] = validation_targets_np
    validation_rows["absolute_error"] = np.abs(
        validation_predictions - validation_targets_np
    )
    validation_path = output_dir / "validation_predictions.csv"
    validation_rows.to_csv(validation_path, index=False)

    configuration = {
        "preprocessed_dir": str(args.preprocessed_dir),
        "shards": [str(path) for path in shard_paths],
        "split_ratio": {"train": 6, "test": 1, "validation": 1},
        "split_sizes": {name: len(value) for name, value in indices.items()},
        "seed": args.seed,
        "device": args.device,
        "input_dim": int(features.shape[1]),
        "num_bins": args.num_bins,
        "target_quantiles": [low_quantile, high_quantile],
        "target_range": list(target_range),
        "loss_type": args.loss_type,
        "lambda_val": args.lambda_val,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": None if args.patience == 0 else args.patience,
        "load_checkpoint": str(args.load_checkpoint) if args.load_checkpoint else None,
    }
    metrics_payload = {
        "test_for_model_selection": test_metrics,
        "final_validation": validation_metrics,
        "history": history,
    }
    record: dict[str, object] = {
        "config": configuration,
        "metrics": metrics_payload,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "split_assignments": str(output_dir / "split_assignments.csv"),
            "validation_predictions": str(validation_path),
        },
    }
    save_json(configuration, output_dir / "manifest.json")
    save_json(metrics_payload, output_dir / "metrics.json")
    save_json(record, output_dir / "result.json")
    logger.info(
        "Final validation MAE %.4f, RMSE %.4f, R2 %.4f",
        validation_metrics["mae"],
        validation_metrics["rmse"],
        validation_metrics["r2"],
    )
    return record


if __name__ == "__main__":
    main()
