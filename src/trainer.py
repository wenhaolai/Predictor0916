"""
training loop for the soft-label regression head for mlp.MLP class.

Two loss function choices:
1. MAE loss: Loss = abs(y_pred-y_label)
2. Combined loss using soft-label regression: Loss = λ·KL(log_softmax(pred_logits)‖y_soft_labels) + (1-λ)·MSE(y_pred, y_label)

Mini-batch training with AdamW.
In-memory history recording for downstream plotting.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .mlp import MLP
from .utils import ensure_dir, get_logger, resolve_device


logger = get_logger(__name__)

class Trainer:
    """Train an ``MLP`` length head with MAE or soft-label regression."""

    SUPPORTED_LOSSES = {"mae", "soft_label"}

    def __init__(
        self,
        model: MLP,
        device: str | torch.device = "auto",
        loss_type: str = "soft_label",
        lambda_val: float = 0.95,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        epochs: int = 10,
        patience: int | None = 3,
        seed: int = 42,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        if loss_type not in self.SUPPORTED_LOSSES:
            raise ValueError(f"loss_type must be one of {sorted(self.SUPPORTED_LOSSES)}")
        if not 0.0 <= lambda_val <= 1.0:
            raise ValueError("lambda_val must lie in [0, 1].")
        if learning_rate <= 0 or batch_size <= 0 or epochs <= 0:
            raise ValueError("learning_rate, batch_size, and epochs must be positive.")
        if patience is not None and patience <= 0:
            raise ValueError("patience must be positive or None.")

        self.model = model
        self.device = resolve_device(device)
        self.loss_type = loss_type
        self.lambda_val = lambda_val
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _loss(
        self,
        logits: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.loss_type == "mae":
            loss = F.l1_loss(predictions, targets)
            return loss, {"mae_loss": float(loss.detach().item())}

        soft_labels = self.model.labels(targets)
        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            soft_labels,
            reduction="batchmean",
        )
        mse_loss = F.mse_loss(predictions, targets)
        loss = self.lambda_val * kl_loss + (1.0 - self.lambda_val) * mse_loss
        return loss, {
            "kl_loss": float(kl_loss.detach().item()),
            "mse_loss": float(mse_loss.detach().item()),
        }

    @staticmethod
    def _validate_tensors(features: torch.Tensor, targets: torch.Tensor) -> None:
        if features.ndim != 2:
            raise ValueError("features must have shape (N, hidden_dim).")
        if targets.reshape(-1).shape[0] != features.shape[0]:
            raise ValueError("features and targets must contain the same number of samples.")
        if features.shape[0] == 0:
            raise ValueError("Training/evaluation tensors must not be empty.")

    def _loader(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        shuffle: bool,
    ) -> DataLoader:
        self._validate_tensors(features, targets)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        dataset = TensorDataset(features.float().cpu(), targets.reshape(-1).float().cpu())
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_absolute_error = 0.0
        sample_count = 0
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits, predictions = self.model(features)
            loss, _ = self._loss(logits, predictions, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = features.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_absolute_error += float(torch.abs(predictions.detach() - targets).sum().item())
            sample_count += batch_size
        return total_loss / sample_count, total_absolute_error / sample_count

    @torch.no_grad()
    def evaluate(self, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Evaluate loss, MAE, and RMSE on a tensor split."""
        loader = self._loader(features, targets, shuffle=False)
        self.model.eval()
        total_loss = 0.0
        sum_absolute_error = 0.0
        sum_squared_error = 0.0
        sample_count = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            logits, predictions = self.model(batch_features)
            loss, _ = self._loss(logits, predictions, batch_targets)
            residual = predictions - batch_targets
            size = batch_features.shape[0]
            total_loss += float(loss.item()) * size
            sum_absolute_error += float(torch.abs(residual).sum().item())
            sum_squared_error += float(torch.square(residual).sum().item())
            sample_count += size
        return {
            "loss": total_loss / sample_count,
            "mae": sum_absolute_error / sample_count,
            "rmse": (sum_squared_error / sample_count) ** 0.5,
        }

    def fit(
        self,
        train_features: torch.Tensor,
        train_targets: torch.Tensor,
        validation_features: torch.Tensor | None = None,
        validation_targets: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Train the head, restore the best state, and return metric history."""
        if (validation_features is None) != (validation_targets is None):
            raise ValueError("validation_features and validation_targets must be provided together.")
        train_loader = self._loader(train_features, train_targets, shuffle=True)
        history: dict[str, Any] = {
            "train_loss": [],
            "train_mae": [],
            "validation_loss": [],
            "validation_mae": [],
            "best_epoch": 0,
            "epochs_trained": 0,
            "stopped_early": False,
        }
        best_score = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        bad_epochs = 0

        for epoch in range(self.epochs):
            train_loss, train_mae = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_mae"].append(train_mae)

            if validation_features is not None and validation_targets is not None:
                validation = self.evaluate(validation_features, validation_targets)
                history["validation_loss"].append(validation["loss"])
                history["validation_mae"].append(validation["mae"])
                score = validation["mae"]
            else:
                score = train_loss

            history["epochs_trained"] = epoch + 1
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(self.model.state_dict())
                history["best_epoch"] = epoch + 1
                bad_epochs = 0
            else:
                bad_epochs += 1

            logger.info(
                "Epoch %d/%d - train_loss=%.6f train_mae=%.4f%s",
                epoch + 1,
                self.epochs,
                train_loss,
                train_mae,
                f" val_mae={score:.4f}" if validation_features is not None else "",
            )
            if self.patience is not None and bad_epochs >= self.patience:
                history["stopped_early"] = True
                break

        self.model.load_state_dict(best_state)
        history["best_score"] = best_score
        if self.checkpoint_dir is not None:
            checkpoint_dir = ensure_dir(self.checkpoint_dir)
            torch.save(self.model.state_dict(), checkpoint_dir / "best_head.pt")
        return history

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict output lengths and return a one-dimensional CPU tensor."""
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must be a non-empty tensor with shape (N, hidden_dim).")
        self.model.eval()
        predictions = []
        for start in range(0, features.shape[0], self.batch_size):
            batch = features[start : start + self.batch_size].float().to(self.device)
            _, batch_predictions = self.model(batch)
            predictions.append(batch_predictions.cpu())
        return torch.cat(predictions, dim=0)
