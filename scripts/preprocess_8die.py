"""Split three raw prompt CSVs and preprocess them on eight two-NPU workers."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PROMPT_COLUMN = "user_prompt_content"
DEFAULT_DEVICE_PAIRS = ";".join(f"{2 * index},{2 * index + 1}" for index in range(8))


def parse_device_pairs(value: str, worker_count: int) -> list[str]:
    """Parse physical NPU pairs such as ``0,1;2,3``."""
    pairs = [pair.strip() for pair in value.split(";") if pair.strip()]
    if len(pairs) != worker_count:
        raise ValueError(
            f"Expected {worker_count} NPU pairs, but received {len(pairs)}: {pairs}"
        )
    normalized = []
    used_devices: set[int] = set()
    for pair in pairs:
        device_ids = [item.strip() for item in pair.split(",")]
        if len(device_ids) != 2 or any(not item.isdigit() for item in device_ids):
            raise ValueError(f"Each worker needs exactly two numeric NPU IDs; got {pair!r}.")
        numeric_ids = [int(item) for item in device_ids]
        overlap = used_devices.intersection(numeric_ids)
        if overlap:
            raise ValueError(f"NPU IDs are assigned to multiple workers: {sorted(overlap)}")
        used_devices.update(numeric_ids)
        normalized.append(",".join(str(item) for item in numeric_ids))
    return normalized


def read_prompt_split(path: Path, split: str) -> pd.DataFrame:
    """Read one raw CSV and attach stable source metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"{split} CSV does not exist: {path}")
    try:
        frame = pd.read_csv(path, usecols=[PROMPT_COLUMN])
    except ValueError as exc:
        raise ValueError(f"{path} must contain column {PROMPT_COLUMN!r}.") from exc
    if frame.empty:
        raise ValueError(f"{split} CSV is empty: {path}")
    frame[PROMPT_COLUMN] = frame[PROMPT_COLUMN].fillna("").astype(str)
    frame.insert(0, "source_index", range(len(frame)))
    frame.insert(0, "source_split", split)
    return frame


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Return a stable fingerprint for a shard's ordered source rows."""
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False):
        digest.update(str(row.source_split).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.source_index).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row.user_prompt_content).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine train/validation/test prompt CSVs, split them into eight shards, "
            "and preprocess all shards concurrently on pairs of Ascend NPUs."
        )
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--model-id-or-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument(
        "--device-pairs",
        default=DEFAULT_DEVICE_PAIRS,
        help='Physical NPU pairs, e.g. "0,1;2,3;...;14,15".',
    )
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device-map", choices=("auto", "balanced", "balanced_low_0", "sequential"), default="balanced")
    parser.add_argument("--max-memory-per-npu", default="48GiB")
    parser.add_argument("--writer-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stagger-seconds", type=float, default=2.0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create shard CSVs and manifest without starting NPU workers.",
    )

    # Internal worker arguments. Users normally do not set these directly.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--visible-devices", help=argparse.SUPPRESS)
    return parser


def run_worker(args: argparse.Namespace) -> int:
    """Load one model replica on two visible NPUs and process one shard."""
    if args.worker_id is None or args.worker_input is None or args.worker_output is None:
        raise ValueError("Worker mode requires worker-id, worker-input, and worker-output.")
    if not args.visible_devices:
        raise ValueError("Worker mode requires visible-devices.")

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.visible_devices
    importlib.import_module("torch_npu")

    from Predictor0916.src.dataset import Dataset
    from Predictor0916.src.model import Model

    started_at = time.time()
    status_path = args.output_dir / "status" / f"shard-{args.worker_id:02d}.json"
    status = {
        "worker_id": args.worker_id,
        "physical_devices": args.visible_devices,
        "input": str(args.worker_input),
        "output": str(args.worker_output),
        "state": "running",
        "started_at_unix": started_at,
    }
    write_json(status, status_path)

    feature_model = Model(
        model_id_or_path=args.model_id_or_path,
        batch_size=args.llm_batch_size,
        device="npu:0",
        torch_dtype=args.torch_dtype,
        max_prompt_length=args.max_prompt_length,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        max_memory={0: args.max_memory_per_npu, 1: args.max_memory_per_npu},
    )
    try:
        feature_model.load_model()
        dataset = Dataset(
            model_id_or_path=args.model_id_or_path,
            local_path=args.worker_input,
            save_path=args.worker_output,
            model=feature_model,
            model_batch_size=args.llm_batch_size,
            device="npu:0",
            torch_dtype=args.torch_dtype,
            max_prompt_length=args.max_prompt_length,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + args.worker_id,
        )
        result = dataset.process(
            writer_batch_size=args.writer_batch_size,
            overwrite=args.overwrite,
        )
        status.update(
            state="completed",
            output=str(result),
            elapsed_seconds=time.time() - started_at,
        )
        write_json(status, status_path)
        return 0
    except Exception as exc:
        status.update(
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.time() - started_at,
        )
        write_json(status, status_path)
        raise
    finally:
        feature_model.unload_model()


def worker_command(
    args: argparse.Namespace,
    worker_id: int,
    worker_input: Path,
    worker_output: Path,
    visible_devices: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-id", str(worker_id),
        "--worker-input", str(worker_input),
        "--worker-output", str(worker_output),
        "--visible-devices", visible_devices,
        "--train-csv", str(args.train_csv),
        "--validation-csv", str(args.validation_csv),
        "--test-csv", str(args.test_csv),
        "--model-id-or-path", args.model_id_or_path,
        "--output-dir", str(args.output_dir),
        "--worker-count", str(args.worker_count),
        "--device-pairs", args.device_pairs,
        "--llm-batch-size", str(args.llm_batch_size),
        "--max-prompt-length", str(args.max_prompt_length),
        "--max-new-tokens", str(args.max_new_tokens),
        "--torch-dtype", args.torch_dtype,
        "--device-map", args.device_map,
        "--max-memory-per-npu", args.max_memory_per_npu,
        "--writer-batch-size", str(args.writer_batch_size),
        "--seed", str(args.seed),
    ]
    command.append("--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code")
    if args.overwrite:
        command.append("--overwrite")
    return command


def run_master(args: argparse.Namespace) -> int:
    if args.worker_count <= 0:
        raise ValueError("worker-count must be positive.")
    if args.llm_batch_size <= 0 or args.max_prompt_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Batch size and token limits must be positive.")
    if args.writer_batch_size <= 0 or args.stagger_seconds < 0:
        raise ValueError("writer-batch-size must be positive and stagger-seconds non-negative.")
    device_pairs = parse_device_pairs(args.device_pairs, args.worker_count)

    frames = [
        read_prompt_split(args.train_csv, "train"),
        read_prompt_split(args.validation_csv, "validation"),
        read_prompt_split(args.test_csv, "test"),
    ]
    split_sizes = {split: len(frame) for split, frame in zip(("train", "validation", "test"), frames)}
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "global_index", range(len(combined)))
    combined = combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    input_dir = args.output_dir / "inputs"
    processed_dir = args.output_dir / "processed"
    log_dir = args.output_dir / "logs"
    for directory in (input_dir, processed_dir, log_dir, args.output_dir / "status"):
        directory.mkdir(parents=True, exist_ok=True)

    shards = []
    for worker_id in range(args.worker_count):
        shard = combined.iloc[worker_id::args.worker_count].reset_index(drop=True)
        if shard.empty:
            raise ValueError(
                f"Worker {worker_id} received no samples; reduce worker-count below {len(combined)}."
            )
        input_path = input_dir / f"shard-{worker_id:02d}.csv"
        output_path = processed_dir / f"shard-{worker_id:02d}.parquet"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output shard already exists: {output_path}. Use --overwrite to regenerate all shards."
            )
        shard.to_csv(input_path, index=False, lineterminator="\n")
        shards.append(
            {
                "worker_id": worker_id,
                "physical_devices": device_pairs[worker_id],
                "samples": len(shard),
                "split_sizes": {
                    key: int(value)
                    for key, value in shard["source_split"].value_counts().to_dict().items()
                },
                "fingerprint": frame_fingerprint(shard),
                "input": str(input_path),
                "output": str(output_path),
                "log": str(log_dir / f"shard-{worker_id:02d}.log"),
            }
        )

    manifest = {
        "model_id_or_path": args.model_id_or_path,
        "source_files": {
            "train": str(args.train_csv),
            "validation": str(args.validation_csv),
            "test": str(args.test_csv),
        },
        "split_sizes": split_sizes,
        "total_samples": len(combined),
        "worker_count": args.worker_count,
        "llm_batch_size": args.llm_batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "max_memory_per_npu": args.max_memory_per_npu,
        "seed": args.seed,
        "shards": shards,
    }
    write_json(manifest, args.output_dir / "manifest.json")
    print(f"Prepared {len(combined)} prompts in {args.worker_count} shards under {args.output_dir}")
    if args.prepare_only:
        return 0

    processes: list[tuple[int, subprocess.Popen, object]] = []
    for shard in shards:
        worker_id = shard["worker_id"]
        log_handle = Path(shard["log"]).open("w", encoding="utf-8", buffering=1)
        command = worker_command(
            args,
            worker_id,
            Path(shard["input"]),
            Path(shard["output"]),
            shard["physical_devices"],
        )
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parents[2],
        )
        processes.append((worker_id, process, log_handle))
        print(
            f"Started worker {worker_id} on physical NPUs {shard['physical_devices']} "
            f"(PID {process.pid}, log {shard['log']})"
        )
        if args.stagger_seconds:
            time.sleep(args.stagger_seconds)

    failures = []
    for worker_id, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append((worker_id, return_code))
        print(f"Worker {worker_id} exited with code {return_code}")
    if failures:
        raise RuntimeError(f"Preprocessing workers failed: {failures}. Check output-dir/logs.")
    print(f"All {args.worker_count} preprocessing workers completed successfully.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_worker(args) if args.worker else run_master(args)


if __name__ == "__main__":
    raise SystemExit(main())
