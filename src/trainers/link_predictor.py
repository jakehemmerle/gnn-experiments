"""Link prediction training loop for knowledge graph embeddings."""

import time

import torch
from torch_geometric.data import Data
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from ..utils import save_checkpoint, save_results


class LinkPredictionTrainer:
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
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        self.device = device
        self.results_dir = results_dir

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr
        )

        # Create data loader using model's built-in loader
        self.loader = model.loader(
            head_index=train_data.edge_index[0],
            rel_type=train_data.edge_type,
            tail_index=train_data.edge_index[1],
            batch_size=config.training.batch_size,
            shuffle=True
        )

        self.best_val_mrr = 0.0
        self.final_loss = 0.0
        self.training_time = 0.0
        self.k = config.evaluation.get("k", 10)

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
    def evaluate(self, data: Data) -> tuple[float, float, float]:
        """Evaluate model using filtered ranking protocol.

        Returns:
            mean_rank, mrr, hits_at_k
        """
        self.model.eval()

        mean_rank, mrr, hits_at_k = self.model.test(
            head_index=data.edge_index[0].to(self.device),
            rel_type=data.edge_type.to(self.device),
            tail_index=data.edge_index[1].to(self.device),
            batch_size=self.config.evaluation.batch_size,
            k=self.k,
            log=False
        )

        return mean_rank, mrr, hits_at_k

    def train(self, log_interval: int = 50) -> dict:
        """Run full training loop.

        Args:
            log_interval: Print metrics every N epochs

        Returns:
            Dictionary with final results
        """
        epochs = self.config.training.epochs

        start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()
            self.final_loss = loss

            if epoch % log_interval == 0 or epoch == 1:
                val_rank, val_mrr, val_hits = self.evaluate(self.val_data)
                print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
                      f"Val MRR: {val_mrr:.4f}, Hits@{self.k}: {val_hits:.4f}")

                if val_mrr > self.best_val_mrr:
                    self.best_val_mrr = val_mrr
                    # Save best model
                    save_checkpoint(
                        self.results_dir / "best_checkpoint.pt",
                        self.model, self.optimizer, epoch,
                        {"val_mrr": val_mrr, "val_hits": val_hits}
                    )

        self.training_time = time.perf_counter() - start_time

        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_rank, test_mrr, test_hits = self.evaluate(self.test_data)

        return {
            "final_loss": self.final_loss,
            "test_mean_rank": test_rank,
            "test_mrr": test_mrr,
            "test_hits_at_k": test_hits,
            "k": self.k,
            "best_val_mrr": self.best_val_mrr,
            "training_time_seconds": self.training_time
        }

    def save(self, run_id: str, model_name: str, dataset_name: str) -> None:
        """Save results and checkpoint."""
        test_rank, test_mrr, test_hits = self.evaluate(self.test_data)

        results = {
            "run_id": run_id,
            "model": model_name,
            "dataset": dataset_name,
            "device": str(self.device),
            "hyperparameters": OmegaConf.to_container(self.config),
            "results": {
                "final_loss": self.final_loss,
                "test_mean_rank": float(test_rank),
                "test_mrr": float(test_mrr),
                "test_hits_at_k": float(test_hits),
                "k": self.k,
                "best_val_mrr": self.best_val_mrr,
                "training_time_seconds": self.training_time
            }
        }

        save_results(self.results_dir / "metrics.json", results)
        save_checkpoint(
            self.results_dir / "checkpoint.pt",
            self.model, self.optimizer, self.config.training.epochs,
            {"test_mrr": test_mrr, "test_hits": test_hits}
        )
        OmegaConf.save(self.config, self.results_dir / "config.yaml")
