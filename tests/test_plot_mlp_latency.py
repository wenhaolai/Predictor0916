from __future__ import annotations

import pandas as pd
import pytest

from Predictor0916.scripts.plot_mlp_latency import BATCH_SIZES, load_results, main


def write_results(path, *, multiplier: float) -> None:
    batches = list(BATCH_SIZES)
    pd.DataFrame(
        {
            "device": ["test"] * len(batches),
            "batch_size": batches,
            "average_request_ms": [multiplier / batch for batch in batches],
            "average_batch_ms": [multiplier * (1 + batch / 16) for batch in batches],
            "throughput_requests_per_second": [batch * 100 / multiplier for batch in batches],
        }
    ).to_csv(path, index=False)


def test_plot_latency_results_writes_pdf_and_png(tmp_path):
    cpu_path = tmp_path / "cpu.csv"
    npu_path = tmp_path / "npu.csv"
    write_results(cpu_path, multiplier=2.0)
    write_results(npu_path, multiplier=1.0)

    pdf_path, png_path = main(
        [
            "--cpu-results",
            str(cpu_path),
            "--npu-results",
            str(npu_path),
            "--output-dir",
            str(tmp_path / "figures"),
            "--output-name",
            "comparison",
            "--title",
            "MLP Predictor Performance",
        ]
    )

    assert pdf_path.is_file()
    assert png_path.is_file()
    assert pdf_path.stat().st_size > 0
    assert png_path.stat().st_size > 0


def test_plot_latency_results_requires_all_batch_sizes(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame(
        {
            "batch_size": [1, 2],
            "average_request_ms": [1.0, 0.5],
            "average_batch_ms": [1.0, 1.0],
            "throughput_requests_per_second": [1.0, 2.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="must contain batch sizes"):
        load_results(path, "CPU")
