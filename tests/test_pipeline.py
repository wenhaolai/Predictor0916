from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from Predictor0916.scripts.test_training import main as training_main
from Predictor0916.src import Dataset, MLP, Model, Trainer


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, prompts, **kwargs):
        assert len(prompts) == 2
        return {
            "input_ids": torch.tensor([[4, 5, 0, 0], [0, 0, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]]),
        }


class FakeBackbone:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        batch, sequence = input_ids.shape
        hidden = torch.zeros(batch, sequence, 3, device=input_ids.device)
        for batch_index in range(batch):
            for position in range(sequence):
                hidden[batch_index, position] = batch_index * 10 + position
        return SimpleNamespace(hidden_states=(hidden,))


class DummyFeatureModel:
    def extract(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        rows = []
        for prompt in prompts:
            value = float(len(prompt))
            rows.append([value, value / 2.0, value % 3.0, 1.0])
        return torch.tensor(rows, dtype=torch.float32)

    def unload_model(self):
        return None


def test_model_extracts_last_non_padding_token_for_both_padding_sides():
    extractor = Model(
        "fake",
        batch_size=2,
        device="cpu",
        torch_dtype="float32",
        model=FakeBackbone(),
        tokenizer=FakeTokenizer(),
    )
    features = extractor.extract(["first", "second"])
    assert features.shape == (2, 3)
    assert torch.equal(features[0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(features[1], torch.tensor([13.0, 13.0, 13.0]))


def test_mlp_forward_and_soft_labels():
    head = MLP(input_dim=8, num_bins=5, target_range=(0.0, 100.0))
    logits, predictions = head(torch.randn(4, 8))
    labels = head.labels(torch.tensor([0.0, 10.0, 55.0, 100.0]))
    assert logits.shape == (4, 5)
    assert predictions.shape == (4,)
    assert labels.shape == (4, 5)
    assert torch.allclose(labels.sum(dim=1), torch.ones(4))
    assert torch.all((predictions >= 0.0) & (predictions <= 100.0))


@pytest.mark.parametrize("loss_type", ["mae", "soft_label"])
def test_trainer_supports_both_losses(loss_type):
    features = torch.randn(20, 6)
    targets = torch.linspace(5.0, 50.0, 20)
    trainer = Trainer(
        MLP(6, num_bins=5, target_range=(0.0, 60.0)),
        device="cpu",
        loss_type=loss_type,
        learning_rate=1e-3,
        batch_size=5,
        epochs=2,
        patience=None,
    )
    history = trainer.fit(features, targets, features[:5], targets[:5])
    predictions = trainer.predict(features[:3])
    assert history["epochs_trained"] == 2
    assert len(history["validation_mae"]) == 2
    assert predictions.shape == (3,)


def test_training_main_runs_end_to_end_and_writes_artifacts(tmp_path, monkeypatch):
    loaded_models = []

    def fake_load_model(model):
        loaded_models.append(model)
        return model

    monkeypatch.setattr(Model, "load_model", fake_load_model)
    data_path = tmp_path / "processed.parquet"
    pd.DataFrame(
        {
            "hidden_state": [[float(index), float(index % 3), 1.0, 0.5] for index in range(12)],
            "response_length": [10 + index * 3 for index in range(12)],
        }
    ).to_parquet(data_path, index=False)
    output_dir = tmp_path / "output"
    record = training_main(
        [
            "--data-path",
            str(data_path),
            "--model-id-or-path",
            "dummy-llm",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--patience",
            "0",
        ]
    )

    assert record["split_sizes"] == {"train": 10, "validation": 2}
    assert record["config"]["model_id_or_path"] == "dummy-llm"
    assert len(loaded_models) == 1
    assert set(record["metrics"]) == {"mae", "rmse", "r2"}
    assert (output_dir / "checkpoints" / "best_layers.pt").exists()
    assert (output_dir / "validation_predictions.csv").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "results.jsonl").exists()


def test_training_main_processes_dataset_when_parquet_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(Model, "load_model", lambda model: model)
    process_calls = []

    def fake_process(dataset):
        process_calls.append(dataset.save_path)
        dataset.save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "hidden_state": [
                    [float(index), float(index % 2), 1.0, 0.5]
                    for index in range(10)
                ],
                "response_length": [8 + index * 2 for index in range(10)],
            }
        ).to_parquet(dataset.save_path, index=False)
        return dataset.save_path

    monkeypatch.setattr("Predictor0916.src.dataset.Dataset.process", fake_process)
    data_path = tmp_path / "missing" / "processed.parquet"
    output_dir = tmp_path / "output"

    training_main(
        [
            "--model-id-or-path",
            "dummy-llm",
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--llm-device",
            "cpu",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--patience",
            "0",
        ]
    )

    assert process_calls == [data_path]
    assert data_path.is_file()


def test_forelen_hub_loader_reads_only_requested_raw_split(tmp_path, monkeypatch):
    raw_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "user_prompt_content": ["first", "second"],
            "response_content": ["a", "b"],
            "target_length": [1, 1],
            "dataset_name": ["example", "example"],
        }
    ).to_csv(raw_csv, index=False)
    download_calls = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        return str(raw_csv)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    dataset = Dataset(subset="qwen2.5-0.5b-rl", source_split="train")
    source = dataset._load_source()

    assert download_calls[0]["filename"] == "qwen2.5_0.5b/RL/train.csv"
    assert list(source.columns) == ["user_prompt_content"]
    assert source["user_prompt_content"].tolist() == ["first", "second"]
