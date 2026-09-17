"""
implement of a light-weight predictor model based on MLP, using regression head for output length prediction

Arch.
-------
Input x: hidden states vector from model.extractor() method, temporarily using the last layers' last token from prompt, 
the shape should be (N, d) where d is hidden_dims of LLM.
Layers: a torch MLP using nn.sequential method, temporarily the stracture is Linear(d, d/2)->Norm->ReLU->Linear(d/2, K).
Outputs: return tow tensors, logits is raw forward from Layers(x), pred is output lengths using pred = /sum softmax(logits)_k*center_k

Dividing output range (min_length, max_length) into K bins with each bin has a center value. The bin centres c_k are computed from the 1st–99th percentile range of training
targets and registered as a non-trainable buffer.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Lightweight MLP head for LLM output length prediction via soft label regression over discretised length bins.
    """
    def __init__(
        self,
        input_dim: int,
        num_bins: int = 20,
        target_range: tuple[float, float] = (0.0, 2048.0),
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2.")
        min_length, max_length = map(float, target_range)
        if not (math.isfinite(min_length) and math.isfinite(max_length)):
            raise ValueError("target_range values must be finite.")
        if max_length <= min_length:
            raise ValueError("target_range must satisfy min_length < max_length.")

        self.input_dim = input_dim
        self.num_bins = num_bins
        self.min_length = min_length
        self.max_length = max_length
        hidden_dim = max(input_dim // 2, 1)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins),
        )

        bin_width = (max_length - min_length) / num_bins
        centers = min_length + (torch.arange(num_bins, dtype=torch.float32) + 0.5) * bin_width
        self.register_buffer("bin_centers", centers)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape (N, {self.input_dim}); got {tuple(inputs.shape)}"
            )
        logits = self.layers(inputs)
        probabilities = F.softmax(logits, dim=-1)
        prediction = torch.sum(probabilities * self.bin_centers.unsqueeze(0), dim=-1)
        return logits, prediction

    def labels(
        self,
        length: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a batch of ground-truth lengths into probability vectors
        over bins via softmax(-|j − i|), where *i* is the ground-truth bin
        of each sample and *j* ranges over 0 … K-1.
        """
        if length.ndim != 1:
            length = length.reshape(-1)
        clipped = torch.clamp(length.float(), self.min_length, self.max_length)
        bin_width = (self.max_length - self.min_length) / self.num_bins
        target_bins = torch.floor((clipped - self.min_length) / bin_width)
        target_bins = torch.clamp(target_bins, 0, self.num_bins - 1).long()
        all_bins = torch.arange(self.num_bins, device=length.device).unsqueeze(0)
        distances = torch.abs(all_bins - target_bins.unsqueeze(1)).float()
        return F.softmax(-distances, dim=-1)

    soft_labels = labels

    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the predicted probability distribution over length bins."""
        logits, _ = self(inputs)
        return F.softmax(logits, dim=-1)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save only the parameters and buffers belonging to ``self.layers``.

        Parent directories are created automatically when they do not exist.

        Parameters
        ----------
        path:
            Destination checkpoint file, for example ``checkpoints/mlp.pt``.
        """
        checkpoint_path = Path(path).expanduser()
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            raise IsADirectoryError(
                f"The checkpoint path must be a file, but received a directory: "
                f"{checkpoint_path}"
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.layers.state_dict(), checkpoint_path)

    def load(
        self,
        path: str | os.PathLike[str],
        *,
        map_location: str | torch.device | None = "cpu",
    ) -> None:
        """Load ``self.layers`` parameters from a checkpoint created by :meth:`save`.

        Parameters
        ----------
        path:
            Existing checkpoint file to load.
        map_location:
            Device mapping passed to :func:`torch.load`. Loading on CPU by
            default makes checkpoints portable across training devices.

        Raises
        ------
        FileNotFoundError
            If no checkpoint exists at ``path``.
        """
        checkpoint_path = Path(path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"MLP layer checkpoint does not exist: {checkpoint_path}"
            )
        if not checkpoint_path.is_file():
            raise IsADirectoryError(
                f"The checkpoint path must be a file, but received: {checkpoint_path}"
            )

        state_dict = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        self.layers.load_state_dict(state_dict, strict=True)
