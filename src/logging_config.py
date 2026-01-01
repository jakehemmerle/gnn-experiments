"""Structured logging configuration for ML experiments.

Provides consistent logging across all modules with support for
both console output and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Default format for log messages
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    name: str = "gnn_experiments"
) -> logging.Logger:
    """Configure and return a logger for ML experiments.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to write logs to file
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "gnn_experiments") -> logging.Logger:
    """Get or create a logger with the given name.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """Structured logger for training metrics.

    Provides consistent formatting for training progress and results.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger()
        self._epoch_metrics: list[dict] = []

    def log_epoch(
        self,
        epoch: int,
        loss: float,
        metrics: dict[str, float],
        prefix: str = ""
    ) -> None:
        """Log metrics for a training epoch.

        Args:
            epoch: Current epoch number
            loss: Training loss
            metrics: Dictionary of metric names to values
            prefix: Optional prefix for the log message
        """
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        msg = f"Epoch {epoch:03d} | Loss: {loss:.4f} | {metrics_str}"
        if prefix:
            msg = f"{prefix} | {msg}"
        self.logger.info(msg)

        # Store for history
        self._epoch_metrics.append({
            "epoch": epoch,
            "loss": loss,
            **metrics
        })

    def log_final_results(self, results: dict[str, float], task: str = "") -> None:
        """Log final training results.

        Args:
            results: Dictionary of final metrics
            task: Task name for context
        """
        self.logger.info("=" * 50)
        self.logger.info(f"Final Results{f' ({task})' if task else ''}")
        self.logger.info("=" * 50)
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

    def get_history(self) -> list[dict]:
        """Get the full training history.

        Returns:
            List of epoch metrics dictionaries
        """
        return self._epoch_metrics.copy()
