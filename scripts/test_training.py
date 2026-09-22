"""Command-line training pipeline for the output-length prediction head.

The expensive LLM inference stage is handled separately by
``src.dataset.Dataset.process``. This script consumes the resulting Parquet
file, trains an MLP head, evaluates it on the validation split, and writes all
experiment artifacts to the selected output directory.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

# When invoked as ``python Predictor0916/scripts/test_training.py``, Python adds
# only the scripts directory to sys.path. Add the repository root so the same
# absolute imports also work for direct execution and ``python -m`` execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Predictor0916.src.dataset import Dataset
from Predictor0916.src.mlp import MLP
from Predictor0916.src.model import Model
from Predictor0916.src.trainer import Trainer
from Predictor0916.src.utils import (
    append_jsonl,
    compute_metrics,
    ensure_dir,
    get_logger,
    save_json,
    set_seed,
)


logger = get_logger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse training options from ``argv`` or the process command line."""
    parser = argparse.ArgumentParser(
        description="Train an MLP output-length predictor from processed ForeLen features."
    )
    parser.add_argument(
        "--model-id-or-path",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Hugging Face model ID or local path of the LLM used for feature extraction.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=1,
        help="Batch size used by the LLM for extraction and generation.",
    )
    parser.add_argument(
        "--llm-device",
        default="auto",
        help="Device used by the LLM, independently of the MLP training device.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Floating-point dtype used to load the LLM.",
    )
    parser.add_argument("--llm-device-map", choices=("auto", "balanced", "balanced_low_0", "sequential"))
    parser.add_argument("--llm-max-memory", nargs="+", metavar="DEVICE=LIMIT",
                        help="Weight budgets, e.g. 0=48GiB 1=48GiB; requires llm-device-map.")
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        help="Optional tokenizer truncation limit for LLM prompts.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of response tokens generated during preprocessing.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom Hugging Face model/tokenizer code.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_DIR / "data" / "llama3.2-1b-rl-generated.parquet",
        help="Parquet file produced by Dataset.process().",
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=None,
        help="Local raw prompt file or dataset directory; used when data-path is missing.",
    )
    parser.add_argument(
        "--dataset-subset",
        default=Dataset.DEFAULT_SUBSET,
        help=(
            "ForeLen config used when data-path does not exist, for example "
            "qwen2.5-0.5b-rl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "test_training",
        help="Directory used for checkpoints, metrics, and predictions.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.2)
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
        help="Early-stopping patience; use 0 to disable early stopping.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        help="Optional self.layers checkpoint created by MLP.save().",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Run data loading, model setup, training, evaluation, and persistence."""
    args = parse_args(argv)
    max_memory = None
    if args.llm_max_memory:
        if args.llm_device_map is None:
            raise ValueError("--llm-max-memory requires --llm-device-map")
        max_memory = {}
        for item in args.llm_max_memory:
            key, separator, limit = item.partition("=")
            if not separator or not limit or not (key.isdigit() or key == "cpu"):
                raise ValueError("Expected memory budgets such as 0=48GiB 1=48GiB")
            max_memory[int(key) if key.isdigit() else key] = limit
    requested_devices = (str(args.llm_device), str(args.device))
    if any(device.split(":", maxsplit=1)[0] == "npu" for device in requested_devices):
        try:
            importlib.import_module("torch_npu")
        except ImportError as exc:
            raise ImportError(
                "NPU execution requires torch_npu. Install the torch_npu version "
                "matching the installed PyTorch and CANN versions."
            ) from exc

    low_quantile, high_quantile = args.target_quantiles
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("target_quantiles must satisfy 0 <= LOW < HIGH <= 1.")
    if args.patience < 0:
        raise ValueError("patience must be non-negative.")

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    logger.info("Loading LLM %s", args.model_id_or_path)
    feature_model = Model(
        model_id_or_path=args.model_id_or_path,
        batch_size=args.llm_batch_size,
        device=args.llm_device,
        torch_dtype=args.torch_dtype,
        max_prompt_length=args.max_prompt_length,
        trust_remote_code=args.trust_remote_code,
        device_map=args.llm_device_map,
        max_memory=max_memory,
    )
    feature_model.load_model()

    logger.info("Loading processed features from %s", args.data_path)
    dataset = Dataset(
        model_id_or_path=args.model_id_or_path,
        subset=args.dataset_subset,
        local_path=args.local_path,
        save_path=args.data_path,
        model=feature_model,
        model_batch_size=args.llm_batch_size,
        device=args.llm_device,
        torch_dtype=args.torch_dtype,
        max_prompt_length=args.max_prompt_length,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    if args.data_path.exists():
        if not args.data_path.is_file():
            raise IsADirectoryError(
                f"Processed dataset path must be a file: {args.data_path}"
            )
        logger.info("Using existing processed dataset: %s", args.data_path)
    else:
        logger.info(
            "Processed dataset does not exist; processing ForeLen into %s",
            args.data_path,
        )
        processed_path = dataset.process()
        if processed_path != args.data_path:
            raise RuntimeError(
                f"Dataset.process() wrote {processed_path}, expected {args.data_path}."
            )

    # The LLM is no longer needed once the processed dataset is available.
    feature_model.unload_model()
    train_loader, validation_loader = dataset.load(
        validation_ratio=args.validation_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=args.drop_last,
    )
    train_features, train_targets = train_loader.dataset.tensors
    validation_features, validation_targets = validation_loader.dataset.tensors

    train_lengths = train_targets.detach().cpu().numpy()
    range_low, range_high = np.quantile(
        train_lengths,
        (low_quantile, high_quantile),
    )
    if range_high <= range_low:
        range_high = range_low + 1.0
    target_range = (float(range_low), float(range_high))

    head = MLP(
        input_dim=int(train_features.shape[1]),
        num_bins=args.num_bins,
        target_range=target_range,
    )
    if args.load_checkpoint is not None:
        logger.info("Loading MLP parameters from %s", args.load_checkpoint)
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
        "Training on %d samples and validating on %d samples",
        len(train_targets),
        len(validation_targets),
    )
    history = trainer.fit(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
    )

    validation_predictions = trainer.predict(validation_features).numpy()
    validation_targets_np = validation_targets.detach().cpu().numpy()
    metrics = compute_metrics(validation_predictions, validation_targets_np)

    checkpoint_path = output_dir / "checkpoints" / "best_layers.pt"
    head.save(checkpoint_path)
    predictions_path = output_dir / "validation_predictions.csv"
    pd.DataFrame(
        {
            "sample_index": np.arange(len(validation_predictions)),
            "predicted_length": validation_predictions,
            "target_length": validation_targets_np,
            "absolute_error": np.abs(validation_predictions - validation_targets_np),
        }
    ).to_csv(predictions_path, index=False)

    configuration = {
        "model_id_or_path": args.model_id_or_path,
        "llm_batch_size": args.llm_batch_size,
        "llm_device": args.llm_device,
        "llm_device_map": args.llm_device_map,
        "llm_max_memory": max_memory,
        "torch_dtype": args.torch_dtype,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "trust_remote_code": args.trust_remote_code,
        "data_path": str(args.data_path),
        "dataset_subset": args.dataset_subset,
        "local_path": str(args.local_path) if args.local_path is not None else None,
        "validation_ratio": args.validation_ratio,
        "input_dim": int(train_features.shape[1]),
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
        "device": args.device,
        "seed": args.seed,
        "load_checkpoint": (
            str(args.load_checkpoint) if args.load_checkpoint is not None else None
        ),
    }
    record: dict[str, object] = {
        "config": configuration,
        "split_sizes": {
            "train": len(train_targets),
            "validation": len(validation_targets),
        },
        "metrics": metrics,
        "history": history,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "predictions": str(predictions_path),
        },
    }
    save_json(configuration, output_dir / "manifest.json")
    save_json({"metrics": metrics, "history": history}, output_dir / "metrics.json")
    append_jsonl([record], output_dir / "results.jsonl")
    logger.info(
        "Validation MAE %.4f, RMSE %.4f, Kendall Tau-b %s",
        metrics["mae"],
        metrics["rmse"],
        (
            f'{metrics["kendall_tau_b"]:.4f}'
            if metrics["kendall_tau_b"] is not None
            else "undefined"
        ),
    )
    return record


if __name__ == "__main__":
    main()
