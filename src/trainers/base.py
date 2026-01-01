"""Base trainer class with common functionality."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, DictConfig

from ..utils import save_checkpoint, save_results
from ..logging_config import TrainingLogger, get_logger


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    Provides common functionality for training, logging, and saving results.
    Subclasses must implement train_epoch() and evaluate() methods.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: DictConfig,
        device: torch.device,
        results_dir: Path
    ):
        self.model = model
        self.config = config
        self.device = device
        self.results_dir = results_dir

        self.optimizer = self._create_optimizer()
        self.logger = TrainingLogger(get_logger())

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.final_loss = 0.0
        self.training_time = 0.0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.get("weight_decay", 0.0)
        )

    @abstractmethod
    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Training loss for the epoch
        """
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Evaluate the model.

        Returns:
            Dictionary of metric names to values
        """
        pass

    @abstractmethod
    def get_primary_metric(self, metrics: dict[str, float]) -> float:
        """Get the primary metric for model selection.

        Args:
            metrics: Dictionary of evaluation metrics

        Returns:
            The primary metric value (higher is better)
        """
        pass

    def train(self, log_interval: int = 20) -> dict[str, Any]:
        """Run full training loop.

        Args:
            log_interval: Log metrics every N epochs

        Returns:
            Dictionary with final results
        """
        epochs = self.config.training.epochs
        start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            loss = self.train_epoch()
            self.final_loss = loss

            if epoch % log_interval == 0 or epoch == 1:
                metrics = self.evaluate()
                self.logger.log_epoch(epoch, loss, metrics)

                primary_metric = self.get_primary_metric(metrics)
                if primary_metric > self.best_metric:
                    self.best_metric = primary_metric
                    self._save_best_checkpoint(metrics)

        self.training_time = time.perf_counter() - start_time

        # Final evaluation
        final_metrics = self.evaluate()
        return self._build_results(final_metrics)

    def _save_best_checkpoint(self, metrics: dict[str, float]) -> None:
        """Save checkpoint for best model."""
        save_checkpoint(
            self.results_dir / "best_checkpoint.pt",
            self.model,
            self.optimizer,
            self.current_epoch,
            metrics
        )

    @abstractmethod
    def _build_results(self, final_metrics: dict[str, float]) -> dict[str, Any]:
        """Build the final results dictionary.

        Args:
            final_metrics: Final evaluation metrics

        Returns:
            Complete results dictionary
        """
        pass

    def save(self, run_id: str, model_name: str, dataset_name: str) -> None:
        """Save results, checkpoint, and config.

        Args:
            run_id: Unique identifier for this run
            model_name: Name of the model
            dataset_name: Name of the dataset
        """
        final_metrics = self.evaluate()

        results = {
            "run_id": run_id,
            "model": model_name,
            "dataset": dataset_name,
            "device": str(self.device),
            "hyperparameters": OmegaConf.to_container(self.config),
            "results": self._build_results(final_metrics)
        }

        save_results(self.results_dir / "metrics.json", results)
        save_checkpoint(
            self.results_dir / "checkpoint.pt",
            self.model,
            self.optimizer,
            self.config.training.epochs,
            final_metrics
        )
        OmegaConf.save(self.config, self.results_dir / "config.yaml")

        self.logger.log_final_results(results["results"])
