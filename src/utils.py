"""
Reproducibility seeds, metric helpers, and logging utilities.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def get_logger(name: str = "predictor0916") -> logging.Logger:
    """Return a consistently configured logger without duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available and CPU otherwise."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def torch_dtype_from_str(dtype: str | torch.dtype) -> torch.dtype:
    """Convert a supported dtype name to a ``torch.dtype``."""
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype {dtype!r}; choose from {sorted(mapping)}") from exc


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a ``Path``."""
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute regression metrics used by the length-prediction experiments."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {predictions.shape} != {targets.shape}"
        )
    if predictions.size == 0:
        raise ValueError("Cannot compute metrics for an empty array.")

    residual = predictions - targets
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    target_variance = float(np.sum(np.square(targets - targets.mean())))
    r2 = 0.0 if target_variance == 0.0 else 1.0 - float(np.sum(np.square(residual))) / target_variance
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(payload: Any, path: str | Path) -> Path:
    """Write a JSON artifact using UTF-8 and stable indentation."""
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
    return output


def append_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Append structured experiment records to a JSON Lines file."""
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return output
