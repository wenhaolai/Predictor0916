"""Benchmark per-request and per-batch latency of the trained MLP predictor.

The timed region contains only repeated ``MLP.forward(batch)`` calls. CSV
parsing, checkpoint loading, model/device setup, warm-up, and the initial
host-to-device copy are deliberately outside the timed region.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Predictor0916.src.mlp import MLP
from Predictor0916.src.utils import ensure_dir, save_json


DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure MLP.forward latency over every hidden-state row in a CSV. "
            "Data loading and host-to-device transfer are excluded from timing."
        )
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Compute device: cpu, npu, or an indexed NPU such as npu:0.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="CSV containing one JSON-encoded hidden-state vector per row.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="MLP layers checkpoint produced by MLP.save().",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for latency_results.csv and latency_results.json.",
    )
    parser.add_argument("--feature-column", default="hidden_state")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        metavar="N",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Untimed forward calls before each batch-size measurement (default: 5).",
    )
    parser.add_argument(
        "--target-range",
        type=float,
        nargs=2,
        default=(0.0, 2048.0),
        metavar=("MIN", "MAX"),
        help=(
            "Range used only to construct MLP bin centers. It does not affect layer "
            "shapes or the latency benchmark (default: 0 2048)."
        ),
    )
    return parser


def normalize_device(device_arg: str) -> torch.device:
    """Validate the requested CPU/NPU platform and return a torch device."""
    normalized = device_arg.strip().lower()
    if normalized == "npu":
        normalized = "npu:0"
    platform = normalized.split(":", maxsplit=1)[0]
    if platform not in {"cpu", "npu"}:
        raise ValueError("--device must be cpu, npu, or an indexed NPU such as npu:0.")
    if platform == "cpu" and normalized != "cpu":
        raise ValueError("CPU does not take an index; use --device cpu.")
    if platform == "npu":
        try:
            importlib.import_module("torch_npu")
        except ImportError as exc:
            raise ImportError(
                "NPU benchmarking requires torch_npu matching the installed "
                "PyTorch and CANN versions."
            ) from exc
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError(f"Requested {normalized}, but no available NPU was found.")
    return torch.device(normalized)


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work; CPU execution is already synchronous."""
    if device.type == "npu":
        torch.npu.synchronize(device)


def load_features(csv_path: Path, feature_column: str) -> torch.Tensor:
    """Load and validate JSON-encoded hidden-state vectors from a CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    try:
        frame = pd.read_csv(csv_path, usecols=[feature_column])
    except ValueError as exc:
        raise ValueError(
            f"CSV {csv_path} does not contain feature column {feature_column!r}."
        ) from exc
    if frame.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    rows: list[np.ndarray] = []
    for row_index, encoded in enumerate(frame[feature_column]):
        try:
            value = json.loads(encoded) if isinstance(encoded, str) else encoded
            row = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not decode {feature_column!r} at CSV row {row_index}: {exc}"
            ) from exc
        if row.ndim != 1 or row.size == 0:
            raise ValueError(
                f"CSV row {row_index} must contain a non-empty 1D feature vector; "
                f"got shape {row.shape}."
            )
        rows.append(row)

    shapes = {row.shape for row in rows}
    if len(shapes) != 1:
        raise ValueError(f"All feature vectors must have the same shape; found {shapes}.")
    matrix = np.stack(rows)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature vectors contain NaN or infinite values.")
    return torch.from_numpy(matrix)


def infer_architecture(checkpoint_path: Path) -> tuple[int, int]:
    """Infer ``input_dim`` and ``num_bins`` from an MLP layers checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint must contain a state dict: {checkpoint_path}")
    try:
        first_weight = state_dict["0.weight"]
        output_weight = state_dict["3.weight"]
    except KeyError as exc:
        raise ValueError(
            "Checkpoint is not an MLP.layers state dict produced by MLP.save(); "
            f"missing key {exc.args[0]!r}."
        ) from exc
    if first_weight.ndim != 2 or output_weight.ndim != 2:
        raise ValueError("Checkpoint Linear weights must be two-dimensional.")
    input_dim = int(first_weight.shape[1])
    num_bins = int(output_weight.shape[0])
    expected_hidden_dim = max(input_dim // 2, 1)
    if tuple(first_weight.shape) != (expected_hidden_dim, input_dim):
        raise ValueError(
            "Checkpoint first layer shape is incompatible with MLP: "
            f"{tuple(first_weight.shape)}."
        )
    if int(output_weight.shape[1]) != expected_hidden_dim:
        raise ValueError(
            "Checkpoint output layer shape is incompatible with MLP: "
            f"{tuple(output_weight.shape)}."
        )
    return input_dim, num_bins


def benchmark_batch_size(
    model: MLP,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
    warmup_batches: int,
) -> dict[str, int | float | str]:
    """Time one complete pass using a single wall-clock interval."""
    sample_count = int(features.shape[0])
    batch_count = math.ceil(sample_count / batch_size)
    warmup_size = min(batch_size, sample_count)

    with torch.inference_mode():
        for _ in range(warmup_batches):
            model.forward(features[:warmup_size])
        synchronize(device)

        start_ns = time.perf_counter_ns()
        for start in range(0, sample_count, batch_size):
            model.forward(features[start : start + batch_size])
        synchronize(device)
        elapsed_ns = time.perf_counter_ns() - start_ns

    total_seconds = elapsed_ns / 1_000_000_000.0
    return {
        "device": str(device),
        "batch_size": batch_size,
        "sample_count": sample_count,
        "batch_count": batch_count,
        "last_batch_size": sample_count - (batch_count - 1) * batch_size,
        "total_seconds": total_seconds,
        "average_request_ms": elapsed_ns / sample_count / 1_000_000.0,
        "average_batch_ms": elapsed_ns / batch_count / 1_000_000.0,
        "throughput_requests_per_second": sample_count / total_seconds,
    }


def main(argv: Sequence[str] | None = None) -> list[dict[str, int | float | str]]:
    args = build_parser().parse_args(argv)
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be non-negative.")
    if not args.batch_sizes or any(size <= 0 for size in args.batch_sizes):
        raise ValueError("Every --batch-sizes value must be positive.")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("--batch-sizes must not contain duplicates.")
    range_low, range_high = map(float, args.target_range)
    if not math.isfinite(range_low) or not math.isfinite(range_high) or range_high <= range_low:
        raise ValueError("--target-range must contain finite MIN < MAX values.")

    device = normalize_device(args.device)
    features = load_features(args.data_path, args.feature_column)
    input_dim, num_bins = infer_architecture(args.checkpoint)
    if int(features.shape[1]) != input_dim:
        raise ValueError(
            f"CSV feature dimension {features.shape[1]} does not match checkpoint "
            f"input dimension {input_dim}."
        )

    model = MLP(
        input_dim=input_dim,
        num_bins=num_bins,
        target_range=(range_low, range_high),
    )
    model.load(args.checkpoint, map_location="cpu")
    model.to(device).eval()
    device_features = features.to(device)

    results = [
        benchmark_batch_size(
            model,
            device_features,
            batch_size,
            device,
            args.warmup_batches,
        )
        for batch_size in args.batch_sizes
    ]

    output_dir = ensure_dir(args.output_dir)
    csv_path = output_dir / "latency_results.csv"
    json_path = output_dir / "latency_results.json"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    save_json(
        {
            "config": {
                "device": str(device),
                "data_path": str(args.data_path),
                "checkpoint": str(args.checkpoint),
                "feature_column": args.feature_column,
                "batch_sizes": list(args.batch_sizes),
                "warmup_batches": args.warmup_batches,
                "timing_scope": "MLP.forward calls only; one interval per full dataset pass",
                "input_preloaded_on_device": True,
                "input_dim": input_dim,
                "num_bins": num_bins,
                "target_range": [range_low, range_high],
            },
            "results": results,
        },
        json_path,
    )

    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    return results


if __name__ == "__main__":
    main()
