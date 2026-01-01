"""Dataset registry for GNN experiments.

Provides a centralized registry for datasets including node classification
and link prediction tasks.
"""

from typing import Callable, Any

from .elliptic import load_elliptic, NodeClassificationData
from .fb15k import load_fb15k237, get_fb15k237_info

# Dataset loaders: name -> (loader_function, task_type)
DATASETS: dict[str, tuple[Callable, str]] = {
    "elliptic": (load_elliptic, "node_classification"),
    "fb15k237": (load_fb15k237, "link_prediction"),
}


def get_dataset(name: str, **kwargs) -> Any:
    """Get a dataset by name.

    Args:
        name: Dataset name (e.g., "elliptic")
        **kwargs: Dataset loader arguments

    Returns:
        Dataset-specific return value (varies by dataset)

    Raises:
        ValueError: If dataset not found
    """
    if name not in DATASETS:
        available = list(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    loader, _ = DATASETS[name]
    return loader(**kwargs)


def get_task_for_dataset(name: str) -> str:
    """Get the task type for a dataset.

    Args:
        name: Dataset name

    Returns:
        Task type ("node_classification" or "link_prediction")
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASETS[name][1]


__all__ = [
    "load_elliptic",
    "load_fb15k237",
    "get_fb15k237_info",
    "get_dataset",
    "get_task_for_dataset",
    "NodeClassificationData",
    "DATASETS",
]
