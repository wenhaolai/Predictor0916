"""Preprocess one large prompt CSV concurrently on eight two-NPU workers."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Predictor0916.scripts.preprocess_8die import (
    DEFAULT_DEVICE_PAIRS,
    PROMPT_COLUMN,
    frame_fingerprint,
    parse_device_pairs,
    run_worker,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read one raw prompt CSV, split it into eight shards, and preprocess "
            "all shards concurrently on pairs of Ascend NPUs."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument(
        "--source-name",
        default="rl-test",
        help="Source label stored in shard sidecars for later provenance tracking.",
    )
    parser.add_argument("--model-id-or-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument(
        "--device-pairs",
        default=DEFAULT_DEVICE_PAIRS,
        help='Physical NPU pairs, e.g. "0,1;2,3;...;14,15".',
    )
    # Keep these defaults aligned with preprocess_8die.py.
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--torch-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-npu", default="48GiB")
    parser.add_argument("--writer-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stagger-seconds", type=float, default=2.0)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create eight input shards and the manifest without starting NPU workers.",
    )

    # Internal arguments used when this script starts a worker subprocess.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--visible-devices", help=argparse.SUPPRESS)
    return parser


def read_prompts(path: Path, source_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    if not source_name.strip():
        raise ValueError("source-name must be non-empty.")
    try:
        frame = pd.read_csv(path, usecols=[PROMPT_COLUMN])
    except ValueError as exc:
        raise ValueError(f"{path} must contain column {PROMPT_COLUMN!r}.") from exc
    if frame.empty:
        raise ValueError(f"Input CSV is empty: {path}")
    frame[PROMPT_COLUMN] = frame[PROMPT_COLUMN].fillna("").astype(str)
    frame.insert(0, "source_index", range(len(frame)))
    frame.insert(0, "source_split", source_name)
    frame.insert(0, "global_index", range(len(frame)))
    return frame


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
        "--input-csv", str(args.input_csv),
        "--source-name", args.source_name,
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
    command.append(
        "--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code"
    )
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
    prompts = read_prompts(args.input_csv, args.source_name)
    shuffled = prompts.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    input_dir = args.output_dir / "inputs"
    processed_dir = args.output_dir / "processed"
    log_dir = args.output_dir / "logs"
    status_dir = args.output_dir / "status"
    for directory in (input_dir, processed_dir, log_dir, status_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shards = []
    for worker_id in range(args.worker_count):
        shard = shuffled.iloc[worker_id::args.worker_count].reset_index(drop=True)
        if shard.empty:
            raise ValueError(
                f"Worker {worker_id} received no samples; reduce worker-count below {len(shuffled)}."
            )
        input_path = input_dir / f"shard-{worker_id:02d}.csv"
        output_path = processed_dir / f"shard-{worker_id:02d}.parquet"
        log_path = log_dir / f"shard-{worker_id:02d}.log"
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
                "fingerprint": frame_fingerprint(shard),
                "input": str(input_path),
                "output": str(output_path),
                "log": str(log_path),
            }
        )

    manifest = {
        "input_csv": str(args.input_csv),
        "source_name": args.source_name,
        "total_samples": len(shuffled),
        "worker_count": args.worker_count,
        "llm_batch_size": args.llm_batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "model_id_or_path": args.model_id_or_path,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "max_memory_per_npu": args.max_memory_per_npu,
        "seed": args.seed,
        "shards": shards,
    }
    write_json(manifest, args.output_dir / "manifest.json")
    print(
        f"Prepared {len(shuffled)} prompts from {args.input_csv} in "
        f"{args.worker_count} shards under {args.output_dir}"
    )
    if args.prepare_only:
        return 0

    processes: list[tuple[int, subprocess.Popen, object]] = []
    for shard in shards:
        worker_id = shard["worker_id"]
        log_handle = Path(shard["log"]).open("w", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            worker_command(
                args,
                worker_id,
                Path(shard["input"]),
                Path(shard["output"]),
                shard["physical_devices"],
            ),
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
