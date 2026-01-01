"""Utility functions for training and evaluation."""

import torch
import numpy as np
import random
import json
from pathlib import Path
from typing import Any


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict
) -> None:
    """Save full training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)


def save_results(path: Path, results: dict[str, Any]) -> None:
    """Save results to JSON file."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
