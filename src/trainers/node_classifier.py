"""Node classification training loop."""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from ..utils import save_checkpoint, save_results


class NodeClassificationTrainer:
    """Trainer for node classification tasks."""

    def __init__(
        self,
        model: torch.nn.Module,
        data: Data,
        train_mask: torch.Tensor,
        test_mask: torch.Tensor,
        config: DictConfig,
        device: torch.device,
        results_dir: Path
    ):
        self.model = model
        self.data = data
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.config = config
        self.device = device
        self.results_dir = results_dir

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay
        )

        self.best_test_acc = 0.0
        self.final_loss = 0.0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(out[self.train_mask], self.data.y[self.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Evaluate model on train and test sets."""
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)

        train_correct = (pred[self.train_mask] == self.data.y[self.train_mask]).sum()
        train_acc = train_correct / self.train_mask.sum()

        test_correct = (pred[self.test_mask] == self.data.y[self.test_mask]).sum()
        test_acc = test_correct / self.test_mask.sum()

        return train_acc.item(), test_acc.item()

    def train(self, log_interval: int = 20) -> dict:
        """Run full training loop.

        Args:
            log_interval: Print metrics every N epochs

        Returns:
            Dictionary with final results
        """
        epochs = self.config.training.epochs

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()
            self.final_loss = loss

            if epoch % log_interval == 0 or epoch == 1:
                train_acc, test_acc = self.evaluate()
                print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}")

                if test_acc > self.best_test_acc:
                    self.best_test_acc = test_acc

        # Final evaluation
        train_acc, test_acc = self.evaluate()

        return {
            "final_loss": self.final_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best_test_acc": self.best_test_acc
        }

    def save(self, run_id: str, model_name: str, dataset_name: str) -> None:
        """Save results and checkpoint."""
        train_acc, test_acc = self.evaluate()

        results = {
            "run_id": run_id,
            "model": model_name,
            "dataset": dataset_name,
            "device": str(self.device),
            "hyperparameters": OmegaConf.to_container(self.config),
            "results": {
                "final_loss": self.final_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "best_test_acc": self.best_test_acc
            }
        }

        save_results(self.results_dir / "metrics.json", results)
        save_checkpoint(
            self.results_dir / "checkpoint.pt",
            self.model, self.optimizer, self.config.training.epochs,
            {"train_acc": train_acc, "test_acc": test_acc}
        )
        OmegaConf.save(self.config, self.results_dir / "config.yaml")
