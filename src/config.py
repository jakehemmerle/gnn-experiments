"""Core configuration utilities."""

import torch
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Any

from .paths import CONFIGS_DIR


def get_device() -> torch.device:
    """Get the best available device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config(config_path: str | Path) -> DictConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        OmegaConf DictConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def generate_run_id(dataset: str, model: str) -> str:
    """Generate timestamped run ID."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{dataset}_{model}_{timestamp}"


# Config schema for validation
CONFIG_SCHEMA = {
    "dataset": {"required": ["name"]},
    "model": {"required": ["name", "hidden_channels"]},
    "training": {"required": ["lr", "epochs"]},
}


def validate_config(config: DictConfig) -> list[str]:
    """Validate configuration against schema.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    for section, requirements in CONFIG_SCHEMA.items():
        if section not in config:
            errors.append(f"Missing required section: {section}")
            continue

        for field in requirements.get("required", []):
            if field not in config[section]:
                errors.append(f"Missing required field: {section}.{field}")

    return errors


def get_config_defaults() -> dict[str, Any]:
    """Get default configuration values.

    Returns:
        Dictionary of default configuration values
    """
    return {
        "seed": 42,
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        },
        "model": {
            "dropout": 0.5,
            "out_channels": 2,
        },
        "training": {
            "weight_decay": 5e-4,
            "log_interval": 20,
        },
        "evaluation": {
            "batch_size": 20000,
            "k": 10,
        },
    }


def merge_with_defaults(config: DictConfig) -> DictConfig:
    """Merge configuration with defaults.

    Args:
        config: User configuration

    Returns:
        Configuration with defaults applied for missing values
    """
    defaults = OmegaConf.create(get_config_defaults())
    return OmegaConf.merge(defaults, config)
