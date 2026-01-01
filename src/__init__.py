"""GNN Experiments - Graph Neural Network experimentation framework.

This package provides a structured framework for running GNN experiments
with proper configuration management, logging, and experiment tracking.
"""

from .config import get_device, load_config, generate_run_id, validate_config
from .paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR
from .utils import set_seed, save_checkpoint, save_results
from .logging_config import setup_logging, get_logger

__version__ = "0.2.0"

__all__ = [
    "get_device",
    "load_config",
    "generate_run_id",
    "validate_config",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "set_seed",
    "save_checkpoint",
    "save_results",
    "setup_logging",
    "get_logger",
]
