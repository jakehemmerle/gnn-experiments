"""Model registry for GNN experiments.

Provides a centralized registry for node classification and knowledge graph
embedding models. Epic 2 will extend this with KGE models (TransE, DistMult, RotatE).
"""

from typing import Callable
import torch.nn as nn

from .gcn import GCN
from .gat import GAT
from .graphsage import GraphSAGE

# Node classification models
NODE_CLASSIFICATION_MODELS: dict[str, type[nn.Module]] = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
}

# Link prediction models (Epic 2 will populate this)
LINK_PREDICTION_MODELS: dict[str, Callable] = {}


def get_model(name: str, task: str = "node_classification", **kwargs) -> nn.Module:
    """Get a model by name and task type.

    Args:
        name: Model name (e.g., "GCN", "GAT", "GraphSAGE")
        task: Task type ("node_classification" or "link_prediction")
        **kwargs: Model constructor arguments

    Returns:
        Instantiated model

    Raises:
        ValueError: If model not found for the given task
    """
    if task == "node_classification":
        if name not in NODE_CLASSIFICATION_MODELS:
            available = list(NODE_CLASSIFICATION_MODELS.keys())
            raise ValueError(f"Unknown node classification model: {name}. Available: {available}")
        return NODE_CLASSIFICATION_MODELS[name](**kwargs)

    elif task == "link_prediction":
        if name not in LINK_PREDICTION_MODELS:
            available = list(LINK_PREDICTION_MODELS.keys())
            raise ValueError(f"Unknown link prediction model: {name}. Available: {available}")
        return LINK_PREDICTION_MODELS[name](**kwargs)

    else:
        raise ValueError(f"Unknown task type: {task}. Use 'node_classification' or 'link_prediction'")


__all__ = [
    "GCN",
    "GAT",
    "GraphSAGE",
    "get_model",
    "NODE_CLASSIFICATION_MODELS",
    "LINK_PREDICTION_MODELS",
]
