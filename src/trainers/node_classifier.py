"""Node classification training loop."""

import time

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from ..distributed import is_main_process, barrier, maybe_wrap_ddp
from ..utils import save_checkpoint, save_results


class NodeClassificationTrainer:
    """Trainer for node classification tasks with multi-GPU support."""

    def __init__(
        self,
        model: torch.nn.Module,
        data: Data,
        train_mask: torch.Tensor,
        test_mask: torch.Tensor,
        config: DictConfig,
        device: torch.device,
        results_dir: Path,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data = data
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.config = config
        self.device = device
        self.results_dir = results_dir
        self.world_size = world_size

        # Wrap model in DDP if distributed
        self.model, self._raw_model = maybe_wrap_ddp(model, rank, world_size)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay
        )

        # Create NeighborLoader for mini-batch training
        train_idx = train_mask.nonzero().view(-1)

        # Split training indices across ranks for DDP
        if world_size > 1:
            chunks = train_idx.split(len(train_idx) // world_size + 1)
            train_idx = chunks[rank] if rank < len(chunks) else torch.tensor([], dtype=torch.long)

        # Get batch size from config, default to 1024
        batch_size = config.training.get("batch_size", 1024)
        # Get neighbor sampling config, default to 2-hop with [25, 10] neighbors
        num_neighbors = config.training.get("num_neighbors", [25, 10])

        self.train_loader = NeighborLoader(
            data,
            input_nodes=train_idx,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=True,
        )

        self.best_test_acc = 0.0
        self.final_loss = 0.0
        self.training_time = 0.0

    def train_epoch(self) -> float:
        """Train for one epoch using mini-batch training."""
        self.model.train()
        total_loss = 0
        total_examples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(batch.x, batch.edge_index)
            # Only use predictions for seed nodes (first batch_size nodes)
            out = out[:batch.batch_size]
            y = batch.y[:batch.batch_size]

            loss = F.cross_entropy(out, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size

        return total_loss / total_examples if total_examples > 0 else 0.0

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Evaluate model on train and test sets using full graph."""
        self._raw_model.eval()

        # Move data to device for evaluation
        data = self.data.to(self.device)
        train_mask = self.train_mask.to(self.device)
        test_mask = self.test_mask.to(self.device)

        out = self._raw_model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        train_correct = (pred[train_mask] == data.y[train_mask]).sum()
        train_acc = train_correct / train_mask.sum()

        test_correct = (pred[test_mask] == data.y[test_mask]).sum()
        test_acc = test_correct / test_mask.sum()

        return train_acc.item(), test_acc.item()

    def train(self, log_interval: int = 20) -> dict:
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
                # Synchronize before evaluation
                barrier()

                # Only evaluate and log on main process
                if is_main_process():
                    train_acc, test_acc = self.evaluate()
                    print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}")

                    if test_acc > self.best_test_acc:
                        self.best_test_acc = test_acc

                barrier()

        self.training_time = time.perf_counter() - start_time

        # Final evaluation (main process only)
        barrier()
        if is_main_process():
            train_acc, test_acc = self.evaluate()
        else:
            train_acc, test_acc = 0.0, 0.0

        return {
            "final_loss": self.final_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best_test_acc": self.best_test_acc,
            "training_time_seconds": self.training_time
        }

    def save(self, run_id: str, model_name: str, dataset_name: str) -> None:
        """Save results and checkpoint."""
        train_acc, test_acc = self.evaluate()

        results = {
            "run_id": run_id,
            "model": model_name,
            "dataset": dataset_name,
            "device": str(self.device),
            "world_size": self.world_size,
            "hyperparameters": OmegaConf.to_container(self.config),
            "results": {
                "final_loss": self.final_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "best_test_acc": self.best_test_acc,
                "training_time_seconds": self.training_time
            }
        }

        save_results(self.results_dir / "metrics.json", results)
        save_checkpoint(
            self.results_dir / "checkpoint.pt",
            self._raw_model, self.optimizer, self.config.training.epochs,
            {"train_acc": train_acc, "test_acc": test_acc}
        )
        OmegaConf.save(self.config, self.results_dir / "config.yaml")
