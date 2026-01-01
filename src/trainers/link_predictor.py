"""Link prediction training loop for knowledge graph embeddings."""

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from omegaconf import DictConfig

from .base import BaseTrainer


class LinkPredictionTrainer(BaseTrainer):
    """Trainer for knowledge graph link prediction tasks."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: Data,
        val_data: Data,
        test_data: Data,
        config: DictConfig,
        device: torch.device,
        results_dir: Path
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        super().__init__(model, config, device, results_dir)

        # Create data loader using model's built-in loader
        self.loader = model.loader(
            head_index=train_data.edge_index[0],
            rel_type=train_data.edge_type,
            tail_index=train_data.edge_index[1],
            batch_size=config.training.batch_size,
            shuffle=True
        )

        self.k = config.evaluation.get("k", 10)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer (no weight decay for KGE models)."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.lr
        )

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_examples = 0

        for head, rel, tail in self.loader:
            head = head.to(self.device)
            rel = rel.to(self.device)
            tail = tail.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.loss(head, rel, tail)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * head.size(0)
            total_examples += head.size(0)

        return total_loss / total_examples

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate model on validation set using filtered ranking protocol."""
        self.model.eval()

        mean_rank, mrr, hits_at_k = self.model.test(
            head_index=self.val_data.edge_index[0].to(self.device),
            rel_type=self.val_data.edge_type.to(self.device),
            tail_index=self.val_data.edge_index[1].to(self.device),
            batch_size=self.config.evaluation.batch_size,
            k=self.k,
            log=False
        )

        return {
            "mean_rank": mean_rank,
            "mrr": mrr,
            f"hits@{self.k}": hits_at_k,
        }

    @torch.no_grad()
    def evaluate_test(self) -> dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()

        mean_rank, mrr, hits_at_k = self.model.test(
            head_index=self.test_data.edge_index[0].to(self.device),
            rel_type=self.test_data.edge_type.to(self.device),
            tail_index=self.test_data.edge_index[1].to(self.device),
            batch_size=self.config.evaluation.batch_size,
            k=self.k,
            log=False
        )

        return {
            "test_mean_rank": mean_rank,
            "test_mrr": mrr,
            f"test_hits@{self.k}": hits_at_k,
        }

    def get_primary_metric(self, metrics: dict[str, float]) -> float:
        """Use MRR for model selection."""
        return metrics["mrr"]

    def _build_results(self, final_metrics: dict[str, float]) -> dict[str, Any]:
        """Build the final results dictionary."""
        test_metrics = self.evaluate_test()

        return {
            "final_loss": self.final_loss,
            "val_mrr": final_metrics["mrr"],
            "val_mean_rank": float(final_metrics["mean_rank"]),
            f"val_hits@{self.k}": float(final_metrics[f"hits@{self.k}"]),
            "test_mrr": float(test_metrics["test_mrr"]),
            "test_mean_rank": float(test_metrics["test_mean_rank"]),
            f"test_hits@{self.k}": float(test_metrics[f"test_hits@{self.k}"]),
            "k": self.k,
            "best_val_mrr": self.best_metric,
            "training_time_seconds": self.training_time
        }
