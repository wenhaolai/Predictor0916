from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from Predictor0916.scripts.benchmark_mlp_latency import (
    infer_architecture,
    load_features,
    main,
)
from Predictor0916.src.mlp import MLP


def test_latency_benchmark_runs_all_rows_and_writes_structured_results(tmp_path):
    input_dim = 8
    sample_count = 11
    data_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "hidden_state": [
                json.dumps([float(row + column) for column in range(input_dim)])
                for row in range(sample_count)
            ],
            "response_length": list(range(sample_count)),
        }
    ).to_csv(data_path, index=False)

    checkpoint_path = tmp_path / "checkpoints" / "best_layers.pt"
    MLP(input_dim=input_dim, num_bins=7, target_range=(1.0, 100.0)).save(
        checkpoint_path
    )
    output_dir = tmp_path / "benchmark"

    results = main(
        [
            "--device",
            "cpu",
            "--data-path",
            str(data_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--batch-sizes",
            "1",
            "4",
            "8",
            "--warmup-batches",
            "0",
        ]
    )

    assert [row["batch_size"] for row in results] == [1, 4, 8]
    assert [row["batch_count"] for row in results] == [11, 3, 2]
    assert [row["last_batch_size"] for row in results] == [1, 3, 3]
    assert all(row["sample_count"] == sample_count for row in results)
    assert all(row["total_seconds"] > 0 for row in results)
    for row in results:
        assert row["average_request_ms"] == pytest.approx(
            row["total_seconds"] * 1000 / sample_count
        )
        assert row["average_batch_ms"] == pytest.approx(
            row["total_seconds"] * 1000 / row["batch_count"]
        )

    csv_results = pd.read_csv(output_dir / "latency_results.csv")
    json_results = json.loads((output_dir / "latency_results.json").read_text())
    assert len(csv_results) == 3
    assert json_results["config"]["input_dim"] == input_dim
    assert json_results["config"]["num_bins"] == 7
    assert json_results["config"]["input_preloaded_on_device"] is True


def test_benchmark_loads_features_and_infers_checkpoint_architecture(tmp_path):
    data_path = tmp_path / "features.csv"
    pd.DataFrame({"hidden_state": ["[1,2,3,4]", "[5,6,7,8]"]}).to_csv(
        data_path, index=False
    )
    checkpoint_path = tmp_path / "mlp.pt"
    MLP(4, num_bins=9).save(checkpoint_path)

    features = load_features(data_path, "hidden_state")
    assert features.dtype == torch.float32
    assert features.shape == (2, 4)
    assert infer_architecture(checkpoint_path) == (4, 9)


def test_benchmark_rejects_feature_dimension_mismatch(tmp_path):
    data_path = tmp_path / "features.csv"
    pd.DataFrame({"hidden_state": ["[1,2,3]"]}).to_csv(data_path, index=False)
    checkpoint_path = tmp_path / "mlp.pt"
    MLP(4, num_bins=5).save(checkpoint_path)

    with pytest.raises(ValueError, match="does not match checkpoint"):
        main(
            [
                "--device",
                "cpu",
                "--data-path",
                str(data_path),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(tmp_path / "output"),
                "--warmup-batches",
                "0",
            ]
        )
