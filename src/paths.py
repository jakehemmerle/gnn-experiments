"""Centralized path configuration for the project.

This module provides a single source of truth for all project directories,
eliminating hardcoded paths scattered throughout the codebase.
"""

from pathlib import Path


# Project root is the parent of the src directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Standard project directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def get_data_dir(dataset_name: str | None = None) -> Path:
    """Get the data directory, optionally for a specific dataset.

    Args:
        dataset_name: Optional dataset name to get its specific directory

    Returns:
        Path to the data directory or dataset-specific subdirectory
    """
    if dataset_name:
        return DATA_DIR / dataset_name
    return DATA_DIR


def get_results_dir(run_id: str | None = None) -> Path:
    """Get the results directory, optionally for a specific run.

    Args:
        run_id: Optional run ID to get its specific directory

    Returns:
        Path to the results directory or run-specific subdirectory
    """
    if run_id:
        return RESULTS_DIR / run_id
    return RESULTS_DIR


def ensure_dirs(*paths: Path) -> None:
    """Ensure directories exist.

    Args:
        *paths: Paths to create if they don't exist
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
