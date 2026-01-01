"""Dataset registry for GNN experiments.

Provides a centralized registry for datasets. Epic 2 will extend this
with FB15k-237 for knowledge graph link prediction.
"""

from typing import Callable, Any

from .elliptic import load_elliptic

# Dataset loaders: name -> (loader_function, task_type)
DATASETS: dict[str, tuple[Callable, str]] = {
    "elliptic": (load_elliptic, "node_classification"),
    # Epic 2 will add:
    # "fb15k237": (load_fb15k237, "link_prediction"),
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
    "get_dataset",
    "get_task_for_dataset",
    "DATASETS",
]
