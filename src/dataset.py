"""
Pre-process dataset from EGTP, huggingface link is https://huggingface.co/datasets/abinzzz/ForeLen

This class has two methods:

1. process
    (1) Call .model.Model to create target model inference class as self.model
    (2) Download or Load from local path, read subset 'llama3.2-1b-rl', extract column 'user_prompt_content'
    (3) Wrap prompts with the model chat template and run batched prefill to extract hidden states
    (4) Instead of using column 'target_length', run batched non-thinking generation to get response lengths
    (5) Save hidden_state and response_length in a parquet file for a given save_path

2. load
    (1) Read the parquet file from a given path produced by process which contains columns 'hidden_state' and 'response_length'.
    (2) Split the loaded data into training and validation sets
    (3) Wrap the splits into a PyTorch Dataset
    (4) Create DataLoader objects for training (shuffle=True) and validation (shuffle=False) with configurable batch_size, num_workers, pin_memory, persistent_workers, etc.
    (5) Return the training and validation DataLoaders.
"""

from __future__ import annotations

import os
import random
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .model import Model
from .utils import ensure_dir, get_logger, set_seed


logger = get_logger(__name__)


def _seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic, distinct random seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

class Dataset:
    """Prepare generated ForeLen features and construct training data loaders."""

    DATASET_ID = "abinzzz/ForeLen"
    DEFAULT_SUBSET = "llama3.2-1b-rl"
    PROMPT_COLUMN = "user_prompt_content"
    _SCENARIO_DIRECTORIES = {
        "longseq": "LongSeq",
        "reasoning": "Reasoning",
        "rl": "RL",
    }

    def __init__(
        self,
        model_id_or_path: str = "meta-llama/Llama-3.2-1B-Instruct",
        *,
        subset: str = DEFAULT_SUBSET,
        source_split: str = "train",
        cache_dir: str | Path | None = None,
        local_path: str | Path | None = None,
        save_path: str | Path = "data/llama3.2-1b-rl-generated.parquet",
        model: Model | None = None,
        model_batch_size: int = 1,
        device: str | torch.device = "auto",
        torch_dtype: str | torch.dtype = "bfloat16",
        max_prompt_length: int | None = None,
        trust_remote_code: bool = True,
        max_new_tokens: int = 1024,
        generation_kwargs: Mapping[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        if not subset:
            raise ValueError("subset must be non-empty.")
        if not source_split:
            raise ValueError("source_split must be non-empty.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        self.model_id_or_path = model_id_or_path
        self.subset = subset
        self.source_split = source_split
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.local_path = Path(local_path) if local_path is not None else None
        self.save_path = Path(save_path)
        self.model = model
        self.model_batch_size = model_batch_size
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_prompt_length = max_prompt_length
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = dict(generation_kwargs or {})
        self.seed = seed

    def _get_model(self) -> Model:
        if self.model is None:
            self.model = Model(
                model_id_or_path=self.model_id_or_path,
                batch_size=self.model_batch_size,
                device=self.device,
                torch_dtype=self.torch_dtype,
                max_prompt_length=self.max_prompt_length,
                trust_remote_code=self.trust_remote_code,
            )
        return self.model

    @staticmethod
    def _read_local_file(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".jsonl", ".json"}:
            return pd.read_json(path, lines=suffix == ".jsonl")
        raise ValueError(f"Unsupported local dataset file: {path}")

    def _load_source(self):
        """Load one ForeLen subset split from disk or the Hugging Face Hub."""
        if self.local_path is not None:
            if not self.local_path.exists():
                raise FileNotFoundError(f"local_path does not exist: {self.local_path}")
            if self.local_path.is_file():
                source = self._read_local_file(self.local_path)
            else:
                direct_files = [
                    self.local_path / f"{self.source_split}.parquet",
                    self.local_path / f"{self.source_split}.csv",
                    self.local_path / f"{self.source_split}.jsonl",
                ]
                local_file = next((path for path in direct_files if path.exists()), None)
                if local_file is not None:
                    source = self._read_local_file(local_file)
                else:
                    try:
                        from datasets import Dataset as HFDataset
                        from datasets import DatasetDict, load_from_disk
                    except ImportError as exc:
                        raise ImportError(
                            "Loading a Hugging Face save_to_disk directory requires the "
                            "'datasets' package. Install it with: pip install datasets"
                        ) from exc
                    loaded = load_from_disk(str(self.local_path))
                    if isinstance(loaded, DatasetDict):
                        if self.source_split not in loaded:
                            raise ValueError(
                                f"Split {self.source_split!r} is not present in {self.local_path}."
                            )
                        source = loaded[self.source_split]
                    elif isinstance(loaded, HFDataset):
                        source = loaded
                    else:
                        raise TypeError(f"Unsupported object loaded from {self.local_path}.")
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "Downloading ForeLen requires the 'huggingface_hub' package. "
                    "Install it with: pip install huggingface_hub"
                ) from exc

            source_file = hf_hub_download(
                repo_id=self.DATASET_ID,
                filename=self._hub_source_filename(),
                repo_type="dataset",
                cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
            )
            # Read only the prompt column from the requested split. ForeLen's
            # raw CSV splits currently have inconsistent auxiliary columns
            # (for example, dataset_name appears in some train files). Loading
            # one raw file avoids datasets.load_dataset trying to cast every
            # split to a single incompatible schema.
            source = pd.read_csv(source_file, usecols=[self.PROMPT_COLUMN])

        columns = set(source.columns if isinstance(source, pd.DataFrame) else source.column_names)
        if self.PROMPT_COLUMN not in columns:
            raise ValueError(
                f"ForeLen source is missing column {self.PROMPT_COLUMN!r}; found {sorted(columns)}"
            )
        return source

    def _hub_source_filename(self) -> str:
        """Map a ForeLen config name to its raw CSV path in the Hub repo."""
        if self.source_split not in {"train", "validation", "test"}:
            raise ValueError(
                "source_split must be one of 'train', 'validation', or 'test' "
                "when downloading ForeLen from the Hub."
            )

        scenario_key = next(
            (
                scenario
                for scenario in self._SCENARIO_DIRECTORIES
                if self.subset.endswith(f"-{scenario}")
            ),
            None,
        )
        if scenario_key is None:
            supported = ", ".join(sorted(self._SCENARIO_DIRECTORIES))
            raise ValueError(
                f"Cannot map ForeLen subset {self.subset!r} to a raw file. "
                f"Its name must end with one of: {supported}."
            )

        model_config = self.subset[: -(len(scenario_key) + 1)]
        try:
            model_family, model_size = model_config.rsplit("-", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Invalid ForeLen subset name: {self.subset!r}") from exc
        model_directory = f"{model_family}_{model_size}"
        scenario_directory = self._SCENARIO_DIRECTORIES[scenario_key]
        return f"{model_directory}/{scenario_directory}/{self.source_split}.csv"

    @staticmethod
    def _iter_prompts(source: Any, max_samples: int | None) -> Iterator[str]:
        total = len(source) if max_samples is None else min(len(source), max_samples)
        if isinstance(source, pd.DataFrame):
            values = source.iloc[:total][Dataset.PROMPT_COLUMN]
            for prompt in values:
                yield "" if pd.isna(prompt) else str(prompt)
            return
        for index in range(total):
            prompt = source[index][Dataset.PROMPT_COLUMN]
            yield "" if prompt is None else str(prompt)

    def process(
        self,
        *,
        save_path: str | Path | None = None,
        max_samples: int | None = None,
        writer_batch_size: int = 32,
        overwrite: bool = False,
        log_every: int = 10,
    ) -> Path:
        """
        Generate hidden states and response lengths and save them as Parquet.

        The output contains exactly ``hidden_state`` (a float list) and
        ``response_length`` (an integer generated-token count). It is written
        incrementally to a temporary file and atomically moved into place only
        after every selected prompt succeeds.
        """
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")
        if writer_batch_size <= 0:
            raise ValueError("writer_batch_size must be positive.")
        output_path = Path(save_path) if save_path is not None else self.save_path
        ensure_dir(output_path.parent)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Pass overwrite=True to replace it."
            )

        set_seed(self.seed)
        source = self._load_source()
        prompts = self._iter_prompts(source, max_samples)
        prompt_count = len(source) if max_samples is None else min(len(source), max_samples)
        model = self._get_model()
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        if temporary_path.exists():
            temporary_path.unlink()

        schema = pa.schema(
            [
                pa.field("hidden_state", pa.list_(pa.float32())),
                pa.field("response_length", pa.int64()),
            ]
        )
        writer: pq.ParquetWriter | None = None
        hidden_buffer: list[list[float]] = []
        length_buffer: list[int] = []
        processed = 0
        prompt_progress = tqdm(
            total=prompt_count,
            desc="Processing prompts",
            unit="prompt",
            dynamic_ncols=True,
        )

        def flush() -> None:
            nonlocal writer
            if not hidden_buffer:
                return
            table = pa.Table.from_pydict(
                {
                    "hidden_state": hidden_buffer,
                    "response_length": length_buffer,
                },
                schema=schema,
            )
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, schema, compression="zstd")
            writer.write_table(table)
            hidden_buffer.clear()
            length_buffer.clear()

        try:
            if model.tokenizer is None:
                model.load_model()
            if model.tokenizer is None or not hasattr(model.tokenizer, "apply_chat_template"):
                raise RuntimeError("The loaded tokenizer does not provide apply_chat_template().")

            original_truncation_side = getattr(model.tokenizer, "truncation_side", "right")
            model.tokenizer.truncation_side = "left"
            inference_batch_size = model.batch_size
            try:
                while True:
                    raw_prompts = list(islice(prompts, inference_batch_size))
                    if not raw_prompts:
                        break
                    formatted_prompts = [
                        model.tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        for prompt in raw_prompts
                    ]
                    hidden_states = model.extract(
                        formatted_prompts,
                        add_special_tokens=False,
                    )
                    batch_size = len(formatted_prompts)
                    if hidden_states.ndim != 2 or hidden_states.shape[0] != batch_size:
                        raise RuntimeError(
                            "Model.extract(batch) must return shape (batch_size, hidden_dim)."
                        )
                    response_lengths = model.generate(
                        formatted_prompts,
                        max_new_tokens=self.max_new_tokens,
                        add_special_tokens=False,
                        **self.generation_kwargs,
                    )
                    if response_lengths.numel() != batch_size:
                        raise RuntimeError(
                            "Model.generate(batch) must return one length per prompt."
                        )
                    hidden_buffer.extend(hidden_states.detach().cpu().float().tolist())
                    batch_lengths = [
                        int(value) for value in response_lengths.reshape(-1).cpu().tolist()
                    ]
                    length_buffer.extend(batch_lengths)
                    processed += batch_size
                    prompt_progress.update(batch_size)
                    prompt_progress.set_postfix(
                        batch=batch_size,
                        max_response_length=max(batch_lengths),
                    )
                    if len(hidden_buffer) >= writer_batch_size:
                        flush()
                    if log_every > 0 and (
                        processed % log_every == 0 or processed == prompt_count
                    ):
                        logger.info("Processed %d prompts", processed)
            finally:
                model.tokenizer.truncation_side = original_truncation_side

            flush()
            prompt_progress.close()
            if writer is None:
                raise ValueError("The selected ForeLen source split contains no samples.")
            writer.close()
            writer = None
            os.replace(temporary_path, output_path)
        except Exception:
            prompt_progress.close()
            if writer is not None:
                writer.close()
            if temporary_path.exists():
                temporary_path.unlink()
            raise

        logger.info("Saved %d processed samples to %s", processed, output_path)
        return output_path

    def load(
        self,
        path: str | Path | None = None,
        *,
        validation_ratio: float = 0.2,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        """Load processed Parquet data and return train/validation DataLoaders."""
        if not 0.0 < validation_ratio < 1.0:
            raise ValueError("validation_ratio must lie strictly between 0 and 1.")
        if batch_size <= 0 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative.")
        if prefetch_factor is not None and prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive or None.")

        parquet_path = Path(path) if path is not None else self.save_path
        if not parquet_path.exists():
            raise FileNotFoundError(f"Processed parquet file not found: {parquet_path}")
        frame = pd.read_parquet(parquet_path, columns=["hidden_state", "response_length"])
        if len(frame) < 2:
            raise ValueError("At least two processed samples are required for train/validation split.")

        hidden_rows = [np.asarray(value, dtype=np.float32) for value in frame["hidden_state"]]
        feature_dims = {row.shape for row in hidden_rows}
        if len(feature_dims) != 1 or len(next(iter(feature_dims))) != 1:
            raise ValueError("All hidden_state values must be one-dimensional and equally sized.")
        hidden_states = torch.from_numpy(np.stack(hidden_rows))
        response_lengths_np = frame["response_length"].to_numpy(dtype=np.float32)
        if not np.isfinite(hidden_states.numpy()).all() or not np.isfinite(response_lengths_np).all():
            raise ValueError("Processed dataset contains non-finite values.")
        response_lengths = torch.from_numpy(response_lengths_np)

        active_seed = self.seed if seed is None else seed
        generator = torch.Generator().manual_seed(active_seed)
        permutation = torch.randperm(len(frame), generator=generator)
        validation_size = min(
            len(frame) - 1,
            max(1, int(round(len(frame) * validation_ratio))),
        )
        validation_indices = permutation[:validation_size]
        train_indices = permutation[validation_size:]
        train_dataset = TensorDataset(
            hidden_states[train_indices],
            response_lengths[train_indices],
        )
        validation_dataset = TensorDataset(
            hidden_states[validation_indices],
            response_lengths[validation_indices],
        )

        common_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "worker_init_fn": _seed_worker if num_workers > 0 else None,
        }
        if num_workers > 0:
            common_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                common_kwargs["prefetch_factor"] = prefetch_factor

        train_generator = torch.Generator().manual_seed(active_seed)
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=drop_last,
            generator=train_generator,
            **common_kwargs,
        )
        validation_loader = DataLoader(
            validation_dataset,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )
        return train_loader, validation_loader
