"""Plot CPU/NPU MLP latency benchmark results as a three-panel figure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BATCH_SIZES = (1, 2, 4, 8, 16, 32)
METRICS = (
    ("average_request_ms", "(a) Average request latency", "Latency (ms/request)"),
    ("average_batch_ms", "(b) Average batch latency", "Latency (ms/batch)"),
    (
        "throughput_requests_per_second",
        "(c) Throughput",
        "Throughput (requests/s)",
    ),
)
COLORS = {"CPU": "#0072B2", "NPU": "#D55E00"}
MARKERS = {"CPU": "o", "NPU": "s"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CPU and NPU latency_results.csv files in three horizontal panels."
        )
    )
    parser.add_argument(
        "--cpu-results",
        type=Path,
        required=True,
        help="latency_results.csv produced with --device cpu.",
    )
    parser.add_argument(
        "--npu-results",
        type=Path,
        required=True,
        help="latency_results.csv produced with --device npu or npu:N.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which PDF and PNG figures are written.",
    )
    parser.add_argument(
        "--output-name",
        default="mlp_latency_comparison",
        help="Output filename stem without an extension.",
    )
    parser.add_argument(
        "--title",
        help="Optional title above the full figure.",
    )
    return parser


def load_results(path: Path, platform: str) -> pd.DataFrame:
    """Load one benchmark CSV and enforce a comparable six-batch result set."""
    if not path.is_file():
        raise FileNotFoundError(f"{platform} result CSV does not exist: {path}")

    required_columns = {"batch_size", *(metric[0] for metric in METRICS)}
    frame = pd.read_csv(path)
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{platform} result CSV is missing columns: {sorted(missing_columns)}"
        )
    if frame["batch_size"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["batch_size"].duplicated(False), "batch_size"].tolist()
        )
        raise ValueError(
            f"{platform} result CSV contains duplicate batch sizes: {duplicates}"
        )

    frame = frame.loc[:, sorted(required_columns)].copy()
    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(required_columns)].isna().any().any():
        raise ValueError(f"{platform} result CSV contains non-numeric metric values.")
    if not np.isfinite(frame[list(required_columns)].to_numpy()).all():
        raise ValueError(f"{platform} result CSV contains NaN or infinite values.")

    actual_batches = tuple(sorted(frame["batch_size"].astype(int).tolist()))
    if actual_batches != BATCH_SIZES:
        raise ValueError(
            f"{platform} result CSV must contain batch sizes {BATCH_SIZES}; "
            f"found {actual_batches}."
        )
    metric_columns = [metric[0] for metric in METRICS]
    if (frame[metric_columns] < 0).any().any():
        raise ValueError(f"{platform} result CSV contains negative metric values.")

    return frame.sort_values("batch_size").reset_index(drop=True)


def plot_results(
    cpu_results: pd.DataFrame,
    npu_results: pd.DataFrame,
    output_dir: Path,
    output_name: str,
    title: str | None = None,
) -> tuple[Path, Path]:
    """Create a paper-ready vector PDF and a 300-DPI PNG preview."""
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("--output-name must be a non-empty filename stem.")
    if Path(output_name).suffix:
        raise ValueError("--output-name must not include a file extension.")

    output_dir.mkdir(parents=True, exist_ok=True)
    x_positions = np.arange(len(BATCH_SIZES))
    results_by_platform = {"CPU": cpu_results, "NPU": npu_results}

    rc_parameters = {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc_parameters):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(10.5, 3.25),
            sharex=True,
            constrained_layout=False,
        )

        for axis, (column, panel_title, y_label) in zip(axes, METRICS):
            for platform, frame in results_by_platform.items():
                axis.plot(
                    x_positions,
                    frame[column].to_numpy(),
                    color=COLORS[platform],
                    marker=MARKERS[platform],
                    markersize=4.5,
                    linewidth=1.8,
                    label=platform,
                )
            axis.set_title(panel_title, pad=7)
            axis.set_xlabel("Batch size")
            axis.set_ylabel(y_label)
            axis.set_xticks(x_positions, labels=BATCH_SIZES)
            axis.set_xlim(-0.2, len(BATCH_SIZES) - 0.8)
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.45)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01 if title is None else 0.94),
            ncol=2,
            frameon=False,
            handlelength=2.4,
        )
        if title:
            figure.suptitle(title, fontsize=11, y=1.02)
        figure.subplots_adjust(
            left=0.07,
            right=0.99,
            bottom=0.18,
            top=0.80 if title is None else 0.74,
            wspace=0.34,
        )

        pdf_path = output_dir / f"{output_name}.pdf"
        png_path = output_dir / f"{output_name}.png"
        figure.savefig(pdf_path, bbox_inches="tight")
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

    return pdf_path, png_path


def main(argv: Sequence[str] | None = None) -> tuple[Path, Path]:
    args = build_parser().parse_args(argv)
    cpu_results = load_results(args.cpu_results, "CPU")
    npu_results = load_results(args.npu_results, "NPU")
    pdf_path, png_path = plot_results(
        cpu_results,
        npu_results,
        args.output_dir,
        args.output_name,
        args.title,
    )
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    main()
