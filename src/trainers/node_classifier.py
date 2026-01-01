"""Node classification training loop with validation support."""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from ..datasets.elliptic import NodeClassificationData
from .base import BaseTrainer


class NodeClassificationTrainer(BaseTrainer):
    """Trainer for node classification tasks with train/val/test splits."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: NodeClassificationData,
        config: DictConfig,
        device: torch.device,
        results_dir: Path
    ):
        # Move dataset to device
        self.dataset = dataset.to(device)

        super().__init__(model, config, device, results_dir)

        self.best_val_acc = 0.0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(self.dataset.data.x, self.dataset.data.edge_index)
        loss = F.cross_entropy(
            out[self.dataset.train_mask],
            self.dataset.data.y[self.dataset.train_mask]
        )
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate model on train, val, and test sets."""
        self.model.eval()
        out = self.model(self.dataset.data.x, self.dataset.data.edge_index)
        pred = out.argmax(dim=1)

        def accuracy(mask: torch.Tensor) -> float:
            correct = (pred[mask] == self.dataset.data.y[mask]).sum()
            return (correct / mask.sum()).item()

        return {
            "train_acc": accuracy(self.dataset.train_mask),
            "val_acc": accuracy(self.dataset.val_mask),
            "test_acc": accuracy(self.dataset.test_mask),
        }

    def get_primary_metric(self, metrics: dict[str, float]) -> float:
        """Use validation accuracy for model selection."""
        return metrics["val_acc"]

    def _build_results(self, final_metrics: dict[str, float]) -> dict[str, Any]:
        """Build the final results dictionary."""
        return {
            "final_loss": self.final_loss,
            "train_acc": final_metrics["train_acc"],
            "val_acc": final_metrics["val_acc"],
            "test_acc": final_metrics["test_acc"],
            "best_val_acc": self.best_metric,
            "training_time_seconds": self.training_time
        }
