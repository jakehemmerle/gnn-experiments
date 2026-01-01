"""Core configuration utilities."""

import torch
from datetime import datetime
from omegaconf import OmegaConf
from pathlib import Path


def get_device(rank: int | None = None) -> torch.device:
    """Get the best available device (CUDA or CPU).

    Args:
        rank: Optional GPU rank for distributed training. If provided,
              returns a specific CUDA device. If None, auto-detects.

    Returns:
        torch.device for the appropriate device
    """
    if rank is not None:
        return torch.device(f"cuda:{rank}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    return OmegaConf.load(config_path)


def generate_run_id(dataset: str, model: str) -> str:
    """Generate timestamped run ID."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{dataset}_{model}_{timestamp}"


def ensure_dirs(*paths: Path) -> None:
    """Ensure directories exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
